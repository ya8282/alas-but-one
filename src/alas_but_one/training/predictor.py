import json
import math
from pathlib import Path
from typing import Dict, Optional

from alas_but_one.models.token import Token
from alas_but_one.training.features import FEATURE_NAMES, extract


def check_model_path(path: str) -> None:
    if Path(path).suffix == '.pkl':
        raise ValueError(
            f"{path} is a pickle file, which is no longer supported because "
            "loading one can run arbitrary code. Set model_path to a name ending "
            "in .json in your config, then run alas --train <labels.jsonl> to "
            "regenerate the model."
        )


class MLPredictor:
    """
    Loads a JSON logistic regression model and applies it to token confidence scores.

    If no model file exists at model_path, this is a no-op and the heuristic
    confidence scores from SpellCheckerTask are used instead.
    """

    def __init__(self, model_path: str):
        check_model_path(model_path)
        self.model_path = model_path
        self._model = None
        self._checked = False
        self._problem = None

    @property
    def available(self) -> bool:
        return Path(self.model_path).exists() and self.problem is None

    def _check(self) -> Optional[str]:
        """Returns a user-facing warning if the model cannot be used, else None."""
        path = Path(self.model_path)
        if not path.exists():
            old = path.with_suffix('.pkl')
            if old.exists():
                return (f"Found {old} but not {path}: pickle models are no longer read. "
                        "Run alas --train <labels.jsonl> to create the .json model; "
                        "heuristic scores are used until then.")
            return None
        fix = "Run alas --train <labels.jsonl> to regenerate it; heuristic scores are used until then."
        try:
            with open(path) as f:
                m = json.load(f)
            n = len(FEATURE_NAMES)
            def num(v):
                return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

            def vec(v):
                return isinstance(v, list) and len(v) == n and all(num(x) for x in v)

            ok = (num(m['intercept']) and all(vec(m[k]) for k in ('weights', 'mean', 'scale'))
                  and all(x != 0 for x in m['scale'])
                  and m.get('features', FEATURE_NAMES) == FEATURE_NAMES)
        except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError) as e:
            return f"ML model {path} is unreadable or incomplete ({e!r}). {fix}"
        if not ok:
            return f"ML model {path} does not match the current features. {fix}"
        self._model = m
        return None

    @property
    def problem(self) -> Optional[str]:
        if not self._checked:
            self._problem = self._check()
            self._checked = True
        return self._problem

    def predict_confidence(self, token: Token) -> float:
        """Returns ML-derived typo probability (0.0–1.0) for one token."""
        m = self._model
        z = m['intercept'] + sum(
            w * (x - mu) / sc
            for w, x, mu, sc in zip(m['weights'], extract(token), m['mean'], m['scale'])
        )
        if not math.isfinite(z):
            return token.confidence
        prob = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
        return round(prob, 4)

    def apply(self, token_dict: Dict[str, Token]) -> Dict[str, Token]:
        """Replaces heuristic confidence scores with ML predictions."""
        if not self.available:
            return token_dict

        for token in token_dict.values():
            token.confidence = self.predict_confidence(token)

        return token_dict
