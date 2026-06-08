"""
Time range representation for relaxation time points.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator

from .exceptions import (
    EmptyTimeRangeError,
    InvalidTimeValueError,
    TimeRangeIndexError,
    TimeRangeOverlapError,
)

_TIME_VALUE_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z]+)?\s*$")

_TIME_UNIT_MULTIPLIERS = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "ms": 1e-3,
    "msec": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "us": 1e-6,
    "usec": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
    "ns": 1e-9,
    "nsec": 1e-9,
    "nanosecond": 1e-9,
    "nanoseconds": 1e-9,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}


class TimeValue:
    """
    Represents a time value in seconds.

    Accepts numeric seconds or strings with units (e.g., "100 ms", "1.5s").
    """

    def __init__(self, value: float | int | str | TimeValue):
        if isinstance(value, TimeValue):
            seconds = value.seconds
        elif isinstance(value, (int, float)):
            seconds = float(value)
        elif isinstance(value, str):
            seconds = self._parse_time_string(value)
        else:
            raise InvalidTimeValueError(value, type(value))

        if not math.isfinite(seconds):
            raise InvalidTimeValueError(value, type(value), reason="non-finite time value")

        self._seconds = seconds

    @staticmethod
    def _parse_time_string(value: str) -> float:
        match = _TIME_VALUE_RE.match(value)
        if not match:
            raise InvalidTimeValueError(value, str, reason="invalid time string format")

        number_text, unit_text = match.groups()
        try:
            number = float(number_text)
        except ValueError as exc:
            raise InvalidTimeValueError(value, str, reason="invalid numeric value") from exc

        if not unit_text:
            return number

        unit_key = unit_text.lower()
        if unit_key not in _TIME_UNIT_MULTIPLIERS:
            raise InvalidTimeValueError(
                value,
                str,
                reason=f"unsupported unit '{unit_text}'",
            )

        return number * _TIME_UNIT_MULTIPLIERS[unit_key]

    @property
    def seconds(self) -> float:
        """Return the time value in seconds."""
        return self._seconds

    def __float__(self) -> float:
        return self._seconds

    def __repr__(self) -> str:
        return f"TimeValue({self._seconds:.6g} s)"


class TimeRange:
    """
    Represents a time range in seconds.

    Args:
        low: Lower bound of range (float, str, or TimeValue)
        high: Upper bound of range (float, str, or TimeValue)
    """

    def __init__(
        self,
        low: float | int | str | TimeValue,
        high: float | int | str | TimeValue,
    ):
        if isinstance(low, TimeValue):
            self._low = low
        else:
            self._low = TimeValue(low)

        if isinstance(high, TimeValue):
            self._high = high
        else:
            self._high = TimeValue(high)

        if self._low.seconds == self._high.seconds:
            raise EmptyTimeRangeError(self._low, self._high)

        if self._low.seconds > self._high.seconds:
            self._low, self._high = self._high, self._low

    @property
    def low(self) -> TimeValue:
        """Returns lower bound as TimeValue."""
        return self._low

    @property
    def high(self) -> TimeValue:
        """Returns upper bound as TimeValue."""
        return self._high

    @property
    def low_seconds(self) -> float:
        """Returns lower bound in seconds."""
        return self._low.seconds

    @property
    def high_seconds(self) -> float:
        """Returns upper bound in seconds."""
        return self._high.seconds

    @property
    def width(self) -> float:
        """Returns width of range in seconds."""
        return self._high.seconds - self._low.seconds

    @property
    def center(self) -> float:
        """Returns center of range in seconds."""
        return (self._high.seconds + self._low.seconds) / 2.0

    def contains(self, value: float | int | str | TimeValue) -> bool:
        """
        Check if a value is within this range.

        Args:
            value: Time value to check

        Returns:
            True if value is within range (inclusive)
        """
        if isinstance(value, TimeValue):
            seconds = value.seconds
        elif isinstance(value, str):
            seconds = TimeValue(value).seconds
        else:
            seconds = float(value)

        return self._low.seconds <= seconds <= self._high.seconds

    def overlaps(self, other: TimeRange) -> bool:
        """
        Check if this range overlaps with another.

        Args:
            other: Another TimeRange

        Returns:
            True if ranges overlap
        """
        return not (self._high.seconds < other.low_seconds or self._low.seconds > other.high_seconds)

    def intersection(self, other: TimeRange) -> TimeRange:
        """
        Get intersection with another range.

        Args:
            other: Another TimeRange

        Returns:
            New TimeRange representing intersection

        Raises:
            TimeRangeOverlapError: If ranges don't overlap
        """
        if not self.overlaps(other):
            raise TimeRangeOverlapError(self, other)

        new_low = max(self._low.seconds, other.low_seconds)
        new_high = min(self._high.seconds, other.high_seconds)

        return TimeRange(new_low, new_high)

    def expand(self, margin: float) -> TimeRange:
        """
        Create new range expanded by margin on both sides.

        Args:
            margin: Amount to expand in seconds

        Returns:
            New expanded TimeRange
        """
        return TimeRange(
            self._low.seconds - margin,
            self._high.seconds + margin,
        )

    def to_tuple(self) -> tuple[float, float]:
        """Returns range as (low, high) tuple in seconds."""
        return (self._low.seconds, self._high.seconds)

    def __iter__(self) -> Iterator[float]:
        """Iterate over (low, high) values in seconds."""
        return iter((self._low.seconds, self._high.seconds))

    def __len__(self) -> int:
        """Returns 2 (for unpacking compatibility)."""
        return 2

    def __getitem__(self, index: int) -> float:
        """Get low (0) or high (1) value in seconds."""
        if index == 0:
            return self._low.seconds
        if index == 1:
            return self._high.seconds
        raise TimeRangeIndexError(index)

    def __eq__(self, other) -> bool:
        """Check equality with another TimeRange or tuple."""
        if isinstance(other, TimeRange):
            return (
                abs(self._low.seconds - other.low_seconds) < 1e-9
                and abs(self._high.seconds - other.high_seconds) < 1e-9
            )
        if isinstance(other, tuple) and len(other) == 2:
            return abs(self._low.seconds - other[0]) < 1e-9 and abs(self._high.seconds - other[1]) < 1e-9
        return False

    def __hash__(self) -> int:
        """Return hash of the range."""
        return hash((self._low.seconds, self._high.seconds))

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"TimeRange({self._low.seconds:.6g}, {self._high.seconds:.6g})"

    def __str__(self) -> str:
        """Return string representation."""
        return f"{self._low.seconds:.6g} - {self._high.seconds:.6g} s"
