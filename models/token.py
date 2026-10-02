from dataclasses import dataclass
from typing import Optional
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

    @property
    def uppercase_ratio(self) -> float:
        return self.uppercase_occurrences / len(self.locations) if self.locations else 0.0
