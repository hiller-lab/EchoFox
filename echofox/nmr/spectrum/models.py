from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class _PeakCandidate:
    """Container describing a detected 2D peak candidate in point space."""
    center: Tuple[float, float]
    linewidth_pts: Optional[Tuple[float, float]] = None
    volume: Optional[float] = None
