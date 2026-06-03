from typing import Union
import re

from ..config import config
from .exceptions import (
    InvalidChemicalShiftFormat,
    UnsupportedChemicalShiftType,
    ChemicalShiftRangeError,
)


class ChemicalShift:
    """
    Represents an NMR chemical shift value.

    The ChemicalShift class allows for the creation of a chemical shift object
    using different formats including numeric values (int, float) and string
    representations with 'ppm' units. It provides validation and conversion
    utilities to ensure accurate representation of chemical shift values.

    Examples:
        ChemicalShift(7.26)
        ChemicalShift("7.26 ppm")
        ChemicalShift("7.26ppm")
        ChemicalShift(-0.5)

    Args:
        val: Chemical shift value as int, float, or string (e.g., "7.26 ppm")
        validate_range: If True, validates that the value is within typical NMR range (default from config)
        min_ppm: Minimum allowed chemical shift value (default from config)
        max_ppm: Maximum allowed chemical shift value (default from config)
    """

    def __init__(
        self,
        val: Union[int, float, str],
        validate_range: bool = None,
        min_ppm: float = None,
        max_ppm: float = None,
    ):
        self._min_ppm = min_ppm if min_ppm is not None else config.chemical_shift_min
        self._max_ppm = max_ppm if max_ppm is not None else config.chemical_shift_max
        self._validate_range = (
            validate_range
            if validate_range is not None
            else config.validate_chemical_shifts
        )
        self._ppm = self._convert_to_float(val)

    @property
    def ppm(self) -> float:
        """Returns the chemical shift value in ppm."""
        return self._ppm

    @ppm.setter
    def ppm(self, val: Union[int, float, str]) -> None:
        """Sets the chemical shift value from int, float, or string."""
        self._ppm = self._convert_to_float(val)

    def _convert_to_float(self, value: Union[int, float, str]) -> float:
        """
        Converts various input formats to a float representing ppm.

        Args:
            value: Chemical shift as int, float, or string

        Returns:
            float: Chemical shift in ppm

        Raises:
            InvalidChemicalShiftFormat: If string format is invalid
            UnsupportedChemicalShiftType: If type is not int, float, or str
            ChemicalShiftRangeError: If value is outside allowed range (when validate_range=True)
        """
        if isinstance(value, (int, float)):
            result = float(value)
        elif isinstance(value, str):
            result = self._parse_string(value)
        else:
            raise UnsupportedChemicalShiftType(value, type(value))

        if self._validate_range:
            self._check_range(result)

        return result

    def _parse_string(self, value: str) -> float:
        """
        Parses a string representation of chemical shift.

        Accepts formats like "3.5 ppm", "3.5ppm", "3.5 PPM", or just "3.5"

        Args:
            value: String to parse

        Returns:
            float: Parsed chemical shift value

        Raises:
            InvalidChemicalShiftFormat: If string cannot be parsed
        """
        original_value = value
        value = value.strip()

        # If no 'ppm' suffix, add it for parsing
        if not re.search(r"ppm$", value, re.IGNORECASE):
            value = value + " ppm"

        # Match strings like "3.5 ppm", "3.5ppm", "-0.5 PPM"
        match = re.match(
            r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*ppm$", value, re.IGNORECASE
        )
        if match:
            return float(match.group(1))
        else:
            raise InvalidChemicalShiftFormat(original_value)

    def _check_range(self, value: float) -> None:
        """
        Validates that the chemical shift is within typical NMR range.

        Args:
            value: Chemical shift value to check

        Raises:
            ChemicalShiftRangeError: If value is outside allowed range
        """
        if not (self._min_ppm <= value <= self._max_ppm):
            raise ChemicalShiftRangeError(value, self._min_ppm, self._max_ppm)

    def __str__(self) -> str:
        """Returns string representation of chemical shift."""
        return f"{self._ppm:.3f} ppm"

    def __repr__(self) -> str:
        """Returns detailed representation of chemical shift."""
        return f"ChemicalShift({self._ppm:.3f})"

    def __eq__(self, other) -> bool:
        """Checks equality with another ChemicalShift or numeric value."""
        if isinstance(other, ChemicalShift):
            return abs(self._ppm - other._ppm) < 1e-9
        elif isinstance(other, (int, float)):
            return abs(self._ppm - float(other)) < 1e-9
        return False

    def __lt__(self, other) -> bool:
        """Less than comparison."""
        if isinstance(other, ChemicalShift):
            return self._ppm < other._ppm
        elif isinstance(other, (int, float)):
            return self._ppm < float(other)
        return NotImplemented

    def __le__(self, other) -> bool:
        """Less than or equal comparison."""
        if isinstance(other, ChemicalShift):
            return self._ppm <= other._ppm
        elif isinstance(other, (int, float)):
            return self._ppm <= float(other)
        return NotImplemented

    def __gt__(self, other) -> bool:
        """Greater than comparison."""
        if isinstance(other, ChemicalShift):
            return self._ppm > other._ppm
        elif isinstance(other, (int, float)):
            return self._ppm > float(other)
        return NotImplemented

    def __ge__(self, other) -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, ChemicalShift):
            return self._ppm >= other._ppm
        elif isinstance(other, (int, float)):
            return self._ppm >= float(other)
        return NotImplemented

    def __hash__(self) -> int:
        """Returns hash of the chemical shift value."""
        return hash(self._ppm)

    def __float__(self) -> float:
        """Returns the chemical shift as a float."""
        return self._ppm

    def __int__(self) -> int:
        """Returns the chemical shift as an integer."""
        return int(self._ppm)
