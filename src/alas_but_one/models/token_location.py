from dataclasses import dataclass

@dataclass
class TokenLocation:
    filename: str
    line: int
