from __future__ import annotations

import re
from typing import Optional, Tuple, Union

import numpy as np
from nmrglue.fileio.fileiobase import unit_conversion

from echofox.nmr.chemical_shift import PpmRange
from echofox.core.time import TimeRange, TimeValue
from .exceptions import InvalidDimensionIndexError, InvalidPpmRangeError


def range_bounds(range_obj: Union[PpmRange, TimeRange]) -> Tuple[float, float]:
    if isinstance(range_obj, TimeRange):
        return range_obj.low_seconds, range_obj.high_seconds
    return range_obj.low_ppm, range_obj.high_ppm


def coerce_range_value(
    range_obj: Union[PpmRange, TimeRange],
    value: Union[float, int, str, TimeValue],
) -> float:
    if isinstance(range_obj, TimeRange):
        if isinstance(value, TimeValue):
            return value.seconds
        if isinstance(value, str):
            return TimeValue(value).seconds
        return float(value)
    return float(value)


def init_unit_converters(self) -> None:
    """Initialize unit converters for each dimension."""
    if self._frequencies is None:
        return

    for dim in range(self._ndim):
        if self._unit_converters[dim] is None and self._frequencies[dim]:
            ppm_range = self._dimension_ranges[dim]
            sw_ppm = ppm_range.width
            sw_hz = sw_ppm * self._frequencies[dim]
            car_ppm = ppm_range.center
            car_hz = car_ppm * self._frequencies[dim]

            self._unit_converters[dim] = unit_conversion(
                self._data.shape[dim], True, sw_hz,
                self._frequencies[dim], car_hz,
            )


def get_ppm_axis(self, dimension: int) -> np.ndarray:
    """Get the ppm axis for a specific dimension."""
    if dimension < 0 or dimension >= self._ndim:
        raise InvalidDimensionIndexError(dimension, self._ndim)

    if self._ppm_scales[dimension] is None:
        ppm_range = self._dimension_ranges[dimension]
        n_points = self._data.shape[dimension]
        low, high = range_bounds(ppm_range)
        self._ppm_scales[dimension] = np.linspace(high, low, n_points)

    return self._ppm_scales[dimension]


def get_ppm_scale(self, dimension: int) -> np.ndarray:
    return get_ppm_axis(self, dimension)


def ppm_to_index(self, dimension: int, ppm_value: Union[float, int, str, TimeValue]) -> int:
    if dimension < 0 or dimension >= self._ndim:
        raise InvalidDimensionIndexError(dimension, self._ndim)

    ppm_range = self._dimension_ranges[dimension]
    if not ppm_range.contains(ppm_value):
        low, high = range_bounds(ppm_range)
        raise InvalidPpmRangeError(ppm_value, low, high, dimension)

    if self._unit_converters[dimension] is not None:
        return int(self._unit_converters[dimension](f"{ppm_value} ppm"))

    ppm_axis = get_ppm_axis(self, dimension)
    value = coerce_range_value(ppm_range, ppm_value)
    return int(np.argmin(np.abs(ppm_axis - value)))


def index_to_ppm(self, dimension: int, index: int) -> float:
    if dimension < 0 or dimension >= self._ndim:
        raise InvalidDimensionIndexError(dimension, self._ndim)

    if self._unit_converters[dimension] is not None:
        return self._unit_converters[dimension].ppm(index)

    ppm_axis = get_ppm_axis(self, dimension)
    return float(ppm_axis[index])


def get_nucleus(self, dimension: int) -> Optional[str]:
    if dimension < 0 or dimension >= self._ndim:
        raise InvalidDimensionIndexError(dimension, self._ndim)

    return self._nuclei[dimension] if self._nuclei else None


def get_frequency(self, dimension: int) -> Optional[float]:
    if dimension < 0 or dimension >= self._ndim:
        raise InvalidDimensionIndexError(dimension, self._ndim)

    return self._frequencies[dimension] if self._frequencies else None


def get_label_text(self, dimension: int) -> str:
    if dimension < 0 or dimension >= self._ndim:
        raise InvalidDimensionIndexError(dimension, self._ndim)

    if self._nuclei is None:
        return f'δ (dim {dimension}) [ppm]'

    nucleus = self._nuclei[dimension]
    match = re.match(r"(\d+)([A-Za-z]+)", nucleus)
    if match:
        atomic_number = match.group(1)
        element = match.group(2)
        return f"δ ($^{{{atomic_number}}}${element}) [ppm]"
    return f"δ ({nucleus}) [ppm]"
