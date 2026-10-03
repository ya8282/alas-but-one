import json
import math
from pathlib import Path
from typing import Dict

from alas_but_one.models.token import Token
from alas_but_one.training.features import extract


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

    @property
    def available(self) -> bool:
        return Path(self.model_path).exists()

    def _load(self):
        with open(self.model_path) as f:
            self._model = json.load(f)

    def predict_confidence(self, token: Token) -> float:
        """Returns ML-derived typo probability (0.0–1.0) for one token."""
        try:
            m = self._model
            z = m['intercept'] + sum(
                w * (x - mu) / sc
                for w, x, mu, sc in zip(m['weights'], extract(token), m['mean'], m['scale'])
            )
            prob = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
            return round(prob, 4)
        except Exception:
            return token.confidence

    def apply(self, token_dict: Dict[str, Token]) -> Dict[str, Token]:
        """Replaces heuristic confidence scores with ML predictions."""
        if not self.available:
            return token_dict

        if self._model is None:
            self._load()

        for token in token_dict.values():
            token.confidence = self.predict_confidence(token)

        return token_dict
