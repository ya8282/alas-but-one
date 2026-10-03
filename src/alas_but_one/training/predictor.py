import pickle
from pathlib import Path
from typing import Dict

from alas_but_one.models.token import Token
from alas_but_one.training.features import extract


class MLPredictor:
    """
    Loads a trained sklearn classifier and applies it to token confidence scores.

    If no model file exists at model_path, this is a no-op and the heuristic
    confidence scores from SpellCheckerTask are used instead.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    @property
    def available(self) -> bool:
        return Path(self.model_path).exists()

    def _load(self):
        with open(self.model_path, 'rb') as f:
            self._model = pickle.load(f)

    def predict_confidence(self, token: Token) -> float:
        """Returns ML-derived typo probability (0.0–1.0) for one token."""
        try:
            import numpy as np
            features = np.array([extract(token)])
            prob = self._model.predict_proba(features)[0][1]
            return round(float(prob), 4)
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
