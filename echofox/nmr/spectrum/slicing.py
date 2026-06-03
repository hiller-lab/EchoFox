from __future__ import annotations

import numpy as np

from echofox.core.time import TimeRange
from echofox.nmr.chemical_shift import ChemicalShift, PpmRange

from ..config import config
from . import axes
from .exceptions import (
    DimensionMismatchError,
    InvalidDimensionalityError,
    InvalidDimensionIndexError,
    InvalidPpmRangeError,
    InvalidSliceError,
)


def get_row(self, idx: int | str | float | ChemicalShift) -> np.ndarray:
    if self._ndim != 2:
        raise InvalidDimensionalityError(self._ndim, expected=2)

    if isinstance(idx, int):
        if 0 <= idx < len(self._data):
            return self._data[idx]
        raise InvalidSliceError(f"Row index {idx} out of range [0, {len(self._data)})")

    if isinstance(idx, (str, float)):
        chemical_shift = ChemicalShift(idx)
    elif isinstance(idx, ChemicalShift):
        chemical_shift = idx
    else:
        raise InvalidSliceError(f"Invalid row specification: {idx}")

    row_index = self.ppm_to_index(0, chemical_shift.ppm)
    return self._data[row_index]


def get_column(self, idx: int | str | float | ChemicalShift) -> np.ndarray:
    if self._ndim != 2:
        raise InvalidDimensionalityError(self._ndim, expected=2)

    if isinstance(idx, int):
        if 0 <= idx < len(self._data[0]):
            return self._data[:, idx]
        raise InvalidSliceError(
            f"Column index {idx} out of range [0, {len(self._data[0])})"
        )

    if isinstance(idx, (str, float)):
        chemical_shift = ChemicalShift(idx)
    elif isinstance(idx, ChemicalShift):
        chemical_shift = idx
    else:
        raise InvalidSliceError(f"Invalid column specification: {idx}")

    col_index = self.ppm_to_index(1, chemical_shift.ppm)
    return self._data[:, col_index]


def get_segment(self, min_ppm: float, max_ppm: float, dimension: int = 0) -> np.ndarray:
    if dimension < 0 or dimension >= self._ndim:
        raise InvalidDimensionIndexError(dimension, self._ndim)

    ppm_range = self._dimension_ranges[dimension]
    if not ppm_range.contains(min_ppm):
        low, high = axes.range_bounds(ppm_range)
        raise InvalidPpmRangeError(min_ppm, low, high, dimension)
    if not ppm_range.contains(max_ppm):
        low, high = axes.range_bounds(ppm_range)
        raise InvalidPpmRangeError(max_ppm, low, high, dimension)

    ppm_axis = self.get_ppm_axis(dimension)

    index0 = np.abs(ppm_axis - max_ppm).argmin()
    index1 = np.abs(ppm_axis - min_ppm).argmin()

    if index0 > index1:
        index0, index1 = index1, index0

    if self._ndim == 1:
        return self._data[index0:index1]

    slices = [slice(None)] * self._ndim
    slices[dimension] = slice(index0, index1)
    return self._data[tuple(slices)]


def extract_segment(self, min_ppm: float, max_ppm: float, dimension: int = 0):
    segment_data = self.get_segment(min_ppm, max_ppm, dimension)

    ppm_axis = self.get_ppm_axis(dimension)
    index0 = np.abs(ppm_axis - max_ppm).argmin()
    index1 = np.abs(ppm_axis - min_ppm).argmin()
    if index0 > index1:
        index0, index1 = index1, index0

    new_dimension_ranges = list(self._dimension_ranges)
    segment_ppm_axis = ppm_axis[index0:index1]
    if isinstance(self._dimension_ranges[dimension], TimeRange):
        new_dimension_ranges[dimension] = TimeRange(
            float(segment_ppm_axis.min()),
            float(segment_ppm_axis.max()),
        )
    else:
        new_dimension_ranges[dimension] = PpmRange(
            float(segment_ppm_axis.min()),
            float(segment_ppm_axis.max()),
        )

    new_nuclei = list(self._nuclei) if self._nuclei else None
    new_unit_converters = list(self._unit_converters) if self._unit_converters else None

    return self.__class__(
        data=segment_data,
        dimension_ranges=new_dimension_ranges,
        nuclei=new_nuclei,
        unit_converters=new_unit_converters,
        name=f"{self._name}_segment" if self._name else None,
    )


def extract_subspectrum(
    self,
    dimension_positions: dict,
    tolerance: float | None = None,
):
    sub_data, slice_indices, slice_ppm_actual = get_subspectrum(
        self, dimension_positions, tolerance
    )

    remaining_dims = [d for d in range(self._ndim) if d not in dimension_positions]

    dimension_ranges = [self._dimension_ranges[d] for d in remaining_dims]

    nuclei = None
    if self._nuclei:
        nuclei = [self._nuclei[d] for d in remaining_dims]

    frequencies = None
    if self._frequencies:
        frequencies = [self._frequencies[d] for d in remaining_dims]

    sub_name = self._name or "spectrum"
    position_strs: list[str] = []
    for dim in sorted(dimension_positions.keys()):
        dim_nucleus = self._nuclei[dim] if self._nuclei else f"dim{dim}"
        position_strs.append(f"{dim_nucleus}={dimension_positions[dim]:.2f}")
    sub_name = f"{sub_name}_{'_'.join(position_strs)}"

    sub_spectrum = self.__class__(
        data=sub_data,
        dimension_ranges=dimension_ranges,
        nuclei=nuclei,
        frequencies=frequencies,
        name=sub_name,
        path=self._path,
        slice_indices=slice_indices,
        slice_ppm_actual=slice_ppm_actual,
    )

    new_dim_idx = 0
    for old_dim in remaining_dims:
        if self._unit_converters[old_dim] is not None:
            sub_spectrum._unit_converters[new_dim_idx] = self._unit_converters[old_dim]
        if self._ppm_scales[old_dim] is not None:
            sub_spectrum._ppm_scales[new_dim_idx] = self._ppm_scales[old_dim]
        new_dim_idx += 1

    return sub_spectrum


def extract_trace(
    self,
    trace_dimension: int,
    dimension_positions: float | list[float] | dict,
):
    if self._ndim == 1:
        raise InvalidDimensionalityError(self._ndim, expected="2D or higher")

    if trace_dimension < 0 or trace_dimension >= self._ndim:
        raise InvalidDimensionIndexError(trace_dimension, self._ndim)

    dims_to_slice = [d for d in range(self._ndim) if d != trace_dimension]

    if isinstance(dimension_positions, (int, float)):
        if self._ndim != 2:
            raise DimensionMismatchError(
                "dimension_positions",
                self._ndim - 1,
                1,
            )
        dimension_dict = {dims_to_slice[0]: float(dimension_positions)}

    elif isinstance(dimension_positions, list):
        if len(dimension_positions) != len(dims_to_slice):
            raise DimensionMismatchError(
                "dimension_positions",
                len(dims_to_slice),
                len(dimension_positions),
            )
        dimension_dict = {
            dim: float(ppm) for dim, ppm in zip(dims_to_slice, dimension_positions, strict=True)
        }

    elif isinstance(dimension_positions, dict):
        dimension_dict = {int(k): float(v) for k, v in dimension_positions.items()}
        missing = set(dims_to_slice) - set(dimension_dict.keys())
        if missing:
            raise DimensionMismatchError(
                f"dimension_positions (missing dims: {sorted(missing)})",
                len(dims_to_slice),
                len(dimension_dict),
            )
    else:
        raise InvalidSliceError(
            f"Invalid dimension_positions type: {type(dimension_positions)}"
        )

    trace_data, slice_indices, slice_ppm_actual = get_subspectrum(self, dimension_dict)

    if trace_data.ndim != 1:
        trace_data = trace_data.flatten()

    ppm_range = self._dimension_ranges[trace_dimension]
    nucleus = self._nuclei[trace_dimension] if self._nuclei else None
    frequency = self._frequencies[trace_dimension] if self._frequencies else None

    trace_name = self._name or "spectrum"
    position_strs: list[str] = []
    for dim in sorted(dimension_dict.keys()):
        dim_nucleus = self._nuclei[dim] if self._nuclei else f"dim{dim}"
        position_strs.append(f"{dim_nucleus}={dimension_dict[dim]:.2f}")
    trace_name = f"{trace_name}_trace_{'_'.join(position_strs)}"

    trace_spectrum = self.__class__(
        data=trace_data,
        dimension_ranges=[ppm_range],
        nuclei=[nucleus] if nucleus else None,
        frequencies=[frequency] if frequency else None,
        name=trace_name,
        path=self._path,
        slice_indices=slice_indices,
        slice_ppm_actual=slice_ppm_actual,
    )

    if self._unit_converters[trace_dimension] is not None:
        trace_spectrum._unit_converters[0] = self._unit_converters[trace_dimension]

    if self._ppm_scales[trace_dimension] is not None:
        trace_spectrum._ppm_scales[0] = self._ppm_scales[trace_dimension]

    return trace_spectrum


def get_subspectrum(
    self,
    dimension_positions: dict,
    tolerance: float | None = None,
) -> tuple[np.ndarray, dict, dict]:
    if not dimension_positions:
        raise DimensionMismatchError("dimension_positions", "at least 1", 0)

    if len(dimension_positions) >= self._ndim:
        raise DimensionMismatchError(
            "dimension_positions",
            f"less than {self._ndim}",
            len(dimension_positions),
        )

    for dim in dimension_positions.keys():
        if dim < 0 or dim >= self._ndim:
            raise InvalidDimensionIndexError(dim, self._ndim)

    default_ppm_tolerance = config.default_ppm_tolerance
    default_time_tolerance = config.default_time_tolerance

    slices = [slice(None)] * self._ndim
    slice_indices = {}
    slice_ppm_actual = {}

    for dim, ppm_value in dimension_positions.items():
        dimension_range = self._dimension_ranges[dim]
        ppm_value = axes.coerce_range_value(dimension_range, ppm_value)
        ppm_axis = self.get_ppm_axis(dim)

        idx = int(np.argmin(np.abs(ppm_axis - ppm_value)))
        actual_ppm = float(ppm_axis[idx])

        if tolerance is None:
            if isinstance(dimension_range, TimeRange):
                dim_tolerance = default_time_tolerance
            else:
                nucleus = None
                if self._nuclei and dim < len(self._nuclei):
                    nucleus = str(self._nuclei[dim]).upper()
                if nucleus == "15N":
                    dim_tolerance = config.default_ppm_tolerance_15n
                elif nucleus == "13C":
                    dim_tolerance = config.default_ppm_tolerance_13c
                elif nucleus == "1H":
                    dim_tolerance = config.default_ppm_tolerance_1h
                else:
                    dim_tolerance = default_ppm_tolerance
        else:
            dim_tolerance = tolerance

        if dim_tolerance is not None and abs(actual_ppm - ppm_value) > dim_tolerance:
            unit_label = "s" if isinstance(dimension_range, TimeRange) else "ppm"
            raise InvalidSliceError(
                f"No point within {dim_tolerance} {unit_label} of {ppm_value} in dimension {dim}. "
                f"Closest is {actual_ppm:.4f} {unit_label}"
            )

        slices[dim] = idx
        slice_indices[dim] = idx
        slice_ppm_actual[dim] = actual_ppm

    sub_data = self._data[tuple(slices)]

    return sub_data, slice_indices, slice_ppm_actual
