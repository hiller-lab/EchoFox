class ChemicalShiftError(Exception):
    """Base class for chemical shift-related exceptions."""


class InvalidChemicalShiftError(ChemicalShiftError):
    """Raised when chemical shift value is invalid or cannot be parsed."""

    def __init__(self, value, reason=None):
        if reason:
            message = f"Invalid chemical shift: {value!r} - {reason}"
        else:
            message = f"Invalid chemical shift: {value!r}. Expected format: '<value> ppm' or numeric value"
        super().__init__(message)
        self.value = value


class InvalidChemicalShiftFormat(ChemicalShiftError):
    """Raised when chemical shift string format is invalid."""

    def __init__(self, value):
        message = (
            f"Invalid chemical shift format: {value!r}. Expected format: '<number> ppm' (e.g., '3.5 ppm', '7.2ppm')"
        )
        super().__init__(message)
        self.value = value


class UnsupportedChemicalShiftType(ChemicalShiftError):
    """Raised when an unsupported type is provided for chemical shift."""

    def __init__(self, value, type_received):
        message = (
            f"Unsupported type for chemical shift: {type_received.__name__}. Expected int, float, or str. "
            f"Got: {value!r}"
        )
        super().__init__(message)
        self.value = value
        self.type_received = type_received


class ChemicalShiftRangeError(ChemicalShiftError):
    """Raised when chemical shift value is outside typical NMR range."""

    def __init__(self, value, min_val=None, max_val=None):
        if min_val is not None and max_val is not None:
            message = f"Chemical shift {value} ppm is outside typical range [{min_val}, {max_val}] ppm"
        else:
            message = f"Chemical shift {value} ppm is outside typical NMR range"
        super().__init__(message)
        self.value = value
        self.min_val = min_val
        self.max_val = max_val


"""
Custom exceptions for PpmRange operations.
"""


class PpmRangeError(Exception):
    """Base class for ppm range-related exceptions."""


class InvalidPpmRangeError(PpmRangeError):
    """Raised when ppm range values are invalid."""

    def __init__(self, low, high, reason=None):
        if reason:
            message = f"Invalid ppm range ({low}, {high}): {reason}"
        else:
            message = f"Invalid ppm range: ({low}, {high})"
        super().__init__(message)
        self.low = low
        self.high = high


class PpmRangeValueError(PpmRangeError):
    """Raised when a value is outside the ppm range."""

    def __init__(self, value, low, high):
        message = f"Value {value} ppm is outside range [{low}, {high}] ppm"
        super().__init__(message)
        self.value = value
        self.low = low
        self.high = high


class PpmRangeOverlapError(PpmRangeError):
    """Raised when ppm ranges do not overlap as expected."""

    def __init__(self, range1, range2):
        message = f"Ranges {range1} and {range2} do not overlap"
        super().__init__(message)
        self.range1 = range1
        self.range2 = range2


class EmptyPpmRangeError(PpmRangeError):
    """Raised when ppm range has zero or negative width."""

    def __init__(self, low, high):
        message = f"Empty or invalid ppm range: low ({low}) >= high ({high})"
        super().__init__(message)
        self.low = low
        self.high = high


class PpmRangeIndexError(PpmRangeError):
    """Raised when an invalid index is used to access PpmRange."""

    def __init__(self, index):
        message = f"Index {index} out of range for PpmRange (valid: 0, 1)"
        super().__init__(message)
        self.index = index
