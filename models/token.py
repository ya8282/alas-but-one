from dataclasses import dataclass
from typing import Dict, Optional
from .token_location import TokenLocation


@dataclass
class Token:
    text: str
    repo: str
    locations: list[TokenLocation]
    misspelled: Optional[bool] = None
    confidence: float = 0.0          # 0.0 = not a typo, 1.0 = definitely a typo
    suggestion: Optional[str] = None # best spelling correction
    ignore: str = 'N'
    label: Optional[str] = None      # 'true_positive' | 'false_positive' for training
    ai_reviewed: bool = False
    ai_comment: Optional[str] = None
    uppercase_occurrences: int = 0
    part_occurrences: Optional[Dict[str, int]] = None  # corpus-wide count of the word, or of each component of a hyphenated compound

    @property
    def uppercase_ratio(self) -> float:
        return self.uppercase_occurrences / len(self.locations) if self.locations else 0.0
