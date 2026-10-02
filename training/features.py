from typing import List
from models.token import Token
from spellchecker import SpellChecker

_checker = None


def _get_checker() -> SpellChecker:
    global _checker
    if _checker is None:
        _checker = SpellChecker()
    return _checker


def _edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        dp2 = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp2[j] = dp[j - 1]
            else:
                dp2[j] = 1 + min(dp[j], dp2[j - 1], dp[j - 1])
        dp = dp2
    return dp[n]


FEATURE_NAMES = [
    "is_misspelled",
    "spell_confidence",
    "edit_distance_norm",
    "word_length_norm",
    "has_digits",
    "is_all_upper",
    "is_short",
]


def extract(token: Token) -> List[float]:
    """Returns a fixed-length feature vector for a Token."""
    checker = _get_checker()
    word = token.text

    is_misspelled = 1.0 if token.misspelled else 0.0
    spell_confidence = float(token.confidence)

    edit_dist = 0.0
    if token.misspelled:
        correction = checker.correction(word)
        if correction and correction != word:
            edit_dist = float(_edit_distance(word, correction))

    word_length = min(len(word) / 20.0, 1.0)
    has_digits = 1.0 if any(c.isdigit() for c in word) else 0.0
    is_all_upper = token.uppercase_ratio
    is_short = 1.0 if len(word) <= 2 else 0.0

    return [
        is_misspelled,
        spell_confidence,
        min(edit_dist / 5.0, 1.0),
        word_length,
        has_digits,
        is_all_upper,
        is_short,
    ]
