from typing import Optional, Tuple
from spellchecker import SpellChecker


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


def compute_confidence(
    word: str, misspelled: bool, checker: SpellChecker,
    uppercase_ratio: Optional[float] = None,
) -> Tuple[float, Optional[str]]:
    """
    Returns (confidence, suggestion).

    confidence: 0.0–1.0 probability the word is a real typo worth investigating.
    suggestion: best spelling correction, or None.

    Scoring rationale:
    - Known words start at 0.05 (correctly spelled → unlikely typo)
    - Misspelled words with edit-distance-1 correction → 0.85 (very likely typo)
    - Misspelled with edit-distance-2 → 0.60; distance-3+ → 0.35
    - Misspelled with no good correction → 0.40 (jargon/neologism)
    - Penalties for short words, digit-containing words, acronyms, very long words
    """
    suggestion = None

    if not misspelled:
        score = 0.05
    else:
        correction = checker.correction(word)
        if correction and correction != word:
            suggestion = correction
            dist = _edit_distance(word, correction)
            if dist == 1:
                score = 0.85
            elif dist == 2:
                score = 0.60
            else:
                score = 0.35
        else:
            score = 0.40

    # Patterns that reduce typo likelihood in technical documentation
    if len(word) <= 2:
        score *= 0.4
    if any(c.isdigit() for c in word):
        score *= 0.5
    if uppercase_ratio is None:
        uppercase_ratio = float(word.isupper() and len(word) > 1)
    score *= 1.0 - 0.5 * uppercase_ratio
    if len(word) > 20:
        score *= 0.6   # likely a URL fragment or identifier

    return round(min(1.0, max(0.0, score)), 4), suggestion
