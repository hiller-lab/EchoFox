"""
Custom exceptions for TimeRange operations.
"""


class TimeRangeError(Exception):
    """Base class for time range-related exceptions."""


class InvalidTimeValueError(TimeRangeError):
    """Raised when a time value is invalid or cannot be parsed."""

    def __init__(self, value, type_received, reason=None):
        if reason:
            message = f"Invalid time value: {value!r} - {reason}"
        else:
            message = (
                f"Invalid time value: {value!r}. Expected numeric or string with units"
            )
        super().__init__(message)
        self.value = value
        self.type_received = type_received


class InvalidTimeRangeError(TimeRangeError):
    """Raised when time range values are invalid."""

    def __init__(self, low, high, reason=None):
        if reason:
            message = f"Invalid time range ({low}, {high}): {reason}"
        else:
            message = f"Invalid time range: ({low}, {high})"
        super().__init__(message)
        self.low = low
        self.high = high


class TimeRangeValueError(TimeRangeError):
    """Raised when a value is outside the time range."""

    def __init__(self, value, low, high):
        message = f"Value {value} s is outside range [{low}, {high}] s"
        super().__init__(message)
        self.value = value
        self.low = low
        self.high = high


class TimeRangeOverlapError(TimeRangeError):
    """Raised when time ranges do not overlap as expected."""

    def __init__(self, range1, range2):
        message = f"Ranges {range1} and {range2} do not overlap"
        super().__init__(message)
        self.range1 = range1
        self.range2 = range2


class EmptyTimeRangeError(TimeRangeError):
    """Raised when time range has zero or negative width."""

    def __init__(self, low, high):
        message = f"Empty or invalid time range: low ({low}) >= high ({high})"
        super().__init__(message)
        self.low = low
        self.high = high


class TimeRangeIndexError(TimeRangeError):
    """Raised when an invalid index is used to access TimeRange."""

    def __init__(self, index):
        message = f"Index {index} out of range for TimeRange (valid: 0, 1)"
        super().__init__(message)
        self.index = index
