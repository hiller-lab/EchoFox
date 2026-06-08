from dataclasses import dataclass


@dataclass(frozen=True)
class _PeakCandidate:
    """Container describing a detected 2D peak candidate in point space."""

    center: tuple[float, float]
    linewidth_pts: tuple[float, float] | None = None
    volume: float | None = None
