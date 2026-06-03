"""Custom exceptions for 2D spectrum plotting."""


class Spectrum2DError(Exception):
    """Base exception for 2D spectrum plotting errors."""

    pass


class InvalidSpectrumIndexError(Spectrum2DError):
    """Raised when spectrum_index has an invalid type or value."""

    def __init__(self, value=None):
        if value is not None:
            message = f"spectrum_index must be an integer, list of integers, or 'all', got {type(value).__name__}"
        else:
            message = "spectrum_index must be an integer, list of integers, or 'all'"
        super().__init__(message)


class InvalidAxisError(Spectrum2DError):
    """Raised when an invalid axis is specified."""

    def __init__(self, value=None):
        if value is not None:
            message = f"axis must be 'f1' or 'f2', got '{value}'"
        else:
            message = "axis must be 'f1' or 'f2'"
        super().__init__(message)


class LabelMismatchError(Spectrum2DError):
    """Raised when spectra have inconsistent axis labels."""

    def __init__(self, axis: str, expected: str, actual: str):
        message = f"Inconsistent {axis} labels: expected '{expected}', got '{actual}'"
        super().__init__(message)
        self.axis = axis
        self.expected = expected
        self.actual = actual


class InvalidInsetError(Spectrum2DError):
    """Raised when inset specification is invalid."""

    def __init__(self, inset):
        message = f"Invalid inset specification: {inset}. Expected (excerpt, position, size) or (excerpt, position, size, kwargs)"
        super().__init__(message)
        self.inset = inset
