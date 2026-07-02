from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PitchPoint:
    """
    Represents a single football pitch landmark.
    """

    id: int
    x: float
    y: float
    label: str


