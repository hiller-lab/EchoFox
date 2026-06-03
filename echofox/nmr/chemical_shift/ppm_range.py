"""
PPM Range representation for NMR spectral regions.
"""

from collections.abc import Iterator

from ..config import config
from .chemical_shift import ChemicalShift
from .exceptions import (
    EmptyPpmRangeError,
    PpmRangeIndexError,
    PpmRangeOverlapError,
)


class PpmRange:
    """
    Represents a chemical shift range in ppm.

    Uses ChemicalShift internally for validation and conversion.

    Args:
        low: Lower bound of range (float, str, or ChemicalShift)
        high: Upper bound of range (float, str, or ChemicalShift)
        validate_range: If True, validates chemical shift values (default from config)
    """

    def __init__(
        self,
        low: float | int | str | ChemicalShift,
        high: float | int | str | ChemicalShift,
        validate_range: bool = None,
    ):
        if validate_range is None:
            validate_range = config.validate_chemical_shifts

        # Convert to ChemicalShift if needed
        if isinstance(low, ChemicalShift):
            self._low = low
        else:
            self._low = ChemicalShift(low, validate_range=validate_range)

        if isinstance(high, ChemicalShift):
            self._high = high
        else:
            self._high = ChemicalShift(high, validate_range=validate_range)

        if self._low == self._high:
            raise EmptyPpmRangeError(self)

        # Ensure low <= high
        if self._low.ppm > self._high.ppm:
            self._low, self._high = self._high, self._low

    @property
    def low(self) -> ChemicalShift:
        """Returns lower bound as ChemicalShift."""
        return self._low

    @property
    def high(self) -> ChemicalShift:
        """Returns upper bound as ChemicalShift."""
        return self._high

    @property
    def low_ppm(self) -> float:
        """Returns lower bound in ppm."""
        return self._low.ppm

    @property
    def high_ppm(self) -> float:
        """Returns upper bound in ppm."""
        return self._high.ppm

    @property
    def width(self) -> float:
        """Returns width of range in ppm."""
        return self._high.ppm - self._low.ppm

    @property
    def center(self) -> float:
        """Returns center of range in ppm."""
        return (self._high.ppm + self._low.ppm) / 2

    def contains(self, value: float | int | str | ChemicalShift) -> bool:
        """
        Check if a value is within this range.

        Args:
            value: Chemical shift value to check

        Returns:
            True if value is within range (inclusive)
        """
        if isinstance(value, ChemicalShift):
            ppm = value.ppm
        elif isinstance(value, str):
            ppm = ChemicalShift(value, validate_range=False).ppm
        else:
            ppm = float(value)

        return self._low.ppm <= ppm <= self._high.ppm

    def overlaps(self, other: "PpmRange") -> bool:
        """
        Check if this range overlaps with another.

        Args:
            other: Another PpmRange

        Returns:
            True if ranges overlap
        """
        return not (self._high.ppm < other.low_ppm or self._low.ppm > other.high_ppm)

    def intersection(self, other: "PpmRange") -> "PpmRange":
        """
        Get intersection with another range.

        Args:
            other: Another PpmRange

        Returns:
            New PpmRange representing intersection

        Raises:
            PpmRangeOverlapError: If ranges don't overlap
        """
        if not self.overlaps(other):
            raise PpmRangeOverlapError(self, other)

        new_low = max(self._low.ppm, other.low_ppm)
        new_high = min(self._high.ppm, other.high_ppm)

        return PpmRange(new_low, new_high, validate_range=False)

    def expand(self, margin: float) -> "PpmRange":
        """
        Create new range expanded by margin on both sides.

        Args:
            margin: Amount to expand in ppm

        Returns:
            New expanded PpmRange
        """
        return PpmRange(
            self._low.ppm - margin, self._high.ppm + margin, validate_range=False
        )

    def to_tuple(self) -> tuple[float, float]:
        """Returns range as (low, high) tuple."""
        return (self._low.ppm, self._high.ppm)

    def __iter__(self) -> Iterator[float]:
        """Iterate over (low, high) values."""
        return iter((self._low.ppm, self._high.ppm))

    def __len__(self) -> int:
        """Returns 2 (for unpacking compatibility)."""
        return 2

    def __getitem__(self, index: int) -> float:
        """Get low (0) or high (1) value."""
        if index == 0:
            return self._low.ppm
        elif index == 1:
            return self._high.ppm
        else:
            raise PpmRangeIndexError(index)

    def __eq__(self, other) -> bool:
        """Check equality with another PpmRange or tuple."""
        if isinstance(other, PpmRange):
            return (
                abs(self._low.ppm - other.low_ppm) < 1e-9
                and abs(self._high.ppm - other.high_ppm) < 1e-9
            )
        elif isinstance(other, tuple) and len(other) == 2:
            return (
                abs(self._low.ppm - other[0]) < 1e-9
                and abs(self._high.ppm - other[1]) < 1e-9
            )
        return False

    def __hash__(self) -> int:
        """Return hash of the range."""
        return hash((self._low.ppm, self._high.ppm))

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"PpmRange({self._low.ppm:.3f}, {self._high.ppm:.3f})"

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self._low.ppm:.3f} - {self._high.ppm:.3f} ppm"
