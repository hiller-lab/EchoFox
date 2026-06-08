"""
Custom exceptions for NmrSpectrum operations.
"""


class SpectrumError(Exception):
    """Base class for spectrum-related exceptions."""


class InvalidSpectrumDataError(SpectrumError):
    """Raised when spectrum data is invalid."""

    def __init__(self, reason=None):
        if reason:
            message = f"Invalid spectrum data: {reason}"
        else:
            message = "Invalid spectrum data"
        super().__init__(message)


class InvalidSpectrumFormatError(SpectrumError):
    """Raised when spectrum file format is not supported."""

    def __init__(self, format_name, supported_formats=None):
        if supported_formats:
            message = (
                f"Spectrum format '{format_name}' is not supported. Supported formats: {', '.join(supported_formats)}"
            )
        else:
            message = f"Spectrum format '{format_name}' is not supported."
        super().__init__(message)
        self.format_name = format_name
        self.supported_formats = supported_formats


class SpectrumFileNotFoundError(SpectrumError):
    """Raised when spectrum file or directory cannot be found."""

    def __init__(self, path, file_type="file"):
        message = f"Spectrum {file_type} not found: {path}"
        super().__init__(message)
        self.path = path
        self.file_type = file_type


class CouldNotReadSpectrumError(SpectrumError):
    """Raised when spectrum file cannot be read or parsed."""

    def __init__(self, path, reason=None):
        if reason:
            message = f"Could not read spectrum from '{path}': {reason}"
        else:
            message = f"Could not read spectrum from '{path}'"
        super().__init__(message)
        self.path = path


class InvalidDimensionalityError(SpectrumError):
    """Raised when spectrum dimensionality is invalid or not supported."""

    def __init__(self, ndim, expected=None):
        if expected:
            message = f"Invalid spectrum dimensionality: {ndim}. Expected: {expected}"
        else:
            message = f"Invalid spectrum dimensionality: {ndim}"
        super().__init__(message)
        self.ndim = ndim
        self.expected = expected


class DimensionMismatchError(SpectrumError):
    """Raised when dimensions don't match between data and parameters."""

    def __init__(self, param_name, expected, got):
        message = f"Dimension mismatch for {param_name}: expected {expected}, got {got}"
        super().__init__(message)
        self.param_name = param_name
        self.expected = expected
        self.got = got


class InvalidPpmRangeError(SpectrumError):
    """Raised when ppm range is invalid or out of bounds."""

    def __init__(self, value, min_val=None, max_val=None, dimension=None):
        if min_val is not None and max_val is not None:
            if dimension is not None:
                message = f"ppm value {value} is outside range [{min_val}, {max_val}] for dimension {dimension}"
            else:
                message = f"ppm value {value} is outside range [{min_val}, {max_val}]"
        else:
            message = f"Invalid ppm range: {value}"
        super().__init__(message)
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        self.dimension = dimension


class InvalidDimensionIndexError(SpectrumError):
    """Raised when dimension index is out of range."""

    def __init__(self, index, ndim):
        message = f"Dimension index {index} is out of range for {ndim}D spectrum"
        super().__init__(message)
        self.index = index
        self.ndim = ndim


class NoiseEstimationError(SpectrumError):
    """Raised when noise estimation fails."""

    def __init__(self, method=None, reason=None):
        if method and reason:
            message = f"Noise estimation failed using method '{method}': {reason}"
        elif reason:
            message = f"Noise estimation failed: {reason}"
        else:
            message = "Noise estimation failed"
        super().__init__(message)
        self.method = method


class InvalidSliceError(SpectrumError):
    """Raised when slice parameters are invalid."""

    def __init__(self, reason):
        message = f"Invalid slice: {reason}"
        super().__init__(message)


class UnitConversionError(SpectrumError):
    """Raised when unit conversion fails."""

    def __init__(self, from_unit, to_unit, reason=None):
        if reason:
            message = f"Cannot convert from {from_unit} to {to_unit}: {reason}"
        else:
            message = f"Cannot convert from {from_unit} to {to_unit}"
        super().__init__(message)
        self.from_unit = from_unit
        self.to_unit = to_unit


class PeakPickingError(SpectrumError):
    """Raised when peak picking fails."""

    def __init__(self, reason=None):
        if reason:
            message = f"Peak picking failed: {reason}"
        else:
            message = "Peak picking failed"
        super().__init__(message)


class SpectrumProcessingError(SpectrumError):
    """Raised when spectrum processing operation fails."""

    def __init__(self, operation, reason=None):
        if reason:
            message = f"Spectrum processing '{operation}' failed: {reason}"
        else:
            message = f"Spectrum processing '{operation}' failed"
        super().__init__(message)
        self.operation = operation


class InvalidNucleusError(SpectrumError):
    """Raised when nucleus label is invalid."""

    def __init__(self, nucleus, reason=None):
        if reason:
            message = f"Invalid nucleus '{nucleus}': {reason}"
        else:
            message = f"Invalid nucleus: '{nucleus}'"
        super().__init__(message)
        self.nucleus = nucleus
