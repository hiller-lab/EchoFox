from __future__ import annotations

import numpy as np

from . import axes
from .exceptions import DimensionMismatchError, InvalidDimensionalityError


def get_max_intensity(self) -> float:
    return float(np.max(self._data))


def get_min_intensity(self) -> float:
    return float(np.min(self._data))


def get_intensity_at(self, *ppm_values: float) -> float:
    if len(ppm_values) != self._ndim:
        raise DimensionMismatchError("ppm_values", self._ndim, len(ppm_values))

    indices = tuple(self.ppm_to_index(dim, ppm) for dim, ppm in enumerate(ppm_values))
    return float(self._data[indices])


def extent(self) -> tuple[float, float, float, float]:
    if self._ndim != 2:
        raise InvalidDimensionalityError(self._ndim, expected=2)

    f1_range = self._dimension_ranges[0]
    f2_range = self._dimension_ranges[1]

    f1_low, f1_high = axes.range_bounds(f1_range)
    f2_low, f2_high = axes.range_bounds(f2_range)
    return (f2_high, f2_low, f1_high, f1_low)
