"""
Train a typo-candidate classifier from labeled JSONL output.

Label format in JSONL (set by user after reviewing output):
  "label": "true_positive"   -> word is a real typo
  "label": "false_positive"  -> word is a legitimate term

Usage:
  alas --train path/to/labeled.jsonl
"""
import json
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

from alas_but_one.models.token import Token
from alas_but_one.models.token_location import TokenLocation
from alas_but_one.training.features import extract


def load_labeled_jsonl(path: str) -> Tuple[List[List[float]], List[int]]:
    """Reads a labeled JSONL file and returns (X, y) for sklearn."""
    X, y = [], []
    skipped = 0

    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label = record.get('label')
            if label not in ('true_positive', 'false_positive'):
                skipped += 1
                continue

            locations = [
                TokenLocation(filename=loc['file'], line=loc['line'])
                for loc in record.get('locations', [])
            ]
            token = Token(
                text=record['word'],
                repo=record.get('repo', ''),
                locations=locations,
                misspelled=record.get('misspelled', False),
                confidence=float(record.get('confidence', 0.0)),
                uppercase_occurrences=record.get(
                    'uppercase_occurrences',
                    len(locations) if record['word'].isupper() and len(record['word']) > 1 else 0,
                ),
            )
            X.append(extract(token))
            y.append(1 if label == 'true_positive' else 0)

    if skipped:
        print(f"  Skipped {skipped} unlabeled records.")

    return X, y


def train(jsonl_path: str, model_out: str, min_samples: int = 20) -> bool:
    """
    Train a logistic regression classifier and save it to model_out.
    Returns True on success, False if not enough samples.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        import numpy as np
    except ImportError:
        sys.exit(
            "scikit-learn and numpy are required for training.\n"
            "Run: pip install scikit-learn numpy"
        )

    X, y = load_labeled_jsonl(jsonl_path)

    if len(X) < min_samples:
        print(
            f"Only {len(X)} labeled samples (minimum {min_samples}). "
            "Label more records and try again."
        )
        return False

    X_arr = np.array(X)
    y_arr = np.array(y)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    clf.fit(X_arr, y_arr)

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, 'wb') as f:
        pickle.dump(clf, f)

    n_pos = int(y_arr.sum())
    n_neg = len(y_arr) - n_pos
    print(
        f"Trained on {len(X)} samples ({n_pos} true positives, {n_neg} false positives).\n"
        f"Model saved to {model_out}"
    )
    return True
