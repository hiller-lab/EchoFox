from __future__ import annotations

from typing import Literal

import numpy as np

from .exceptions import InvalidDimensionIndexError, InvalidDimensionalityError, SpectrumProcessingError


def extract_projection(
    self,
    axis: int,
    method: Literal['sum', 'max', 'min', 'mean'] = 'sum',
):
    if self._ndim == 1:
        raise InvalidDimensionalityError(self._ndim, expected="2D or higher")

    projected_data = get_projection(self, axis, method)

    remaining_dims = [d for d in range(self._ndim) if d != axis]

    dimension_ranges = [self._dimension_ranges[d] for d in remaining_dims]

    nuclei = None
    if self._nuclei:
        nuclei = [self._nuclei[d] for d in remaining_dims]

    frequencies = None
    if self._frequencies:
        frequencies = [self._frequencies[d] for d in remaining_dims]

    proj_name = self._name or "spectrum"
    collapsed_nucleus = self._nuclei[axis] if self._nuclei else f"dim{axis}"
    proj_name = f"{proj_name}_proj_{collapsed_nucleus}_{method}"

    proj_spectrum = self.__class__(
        data=projected_data,
        dimension_ranges=dimension_ranges,
        nuclei=nuclei,
        frequencies=frequencies,
        name=proj_name,
        path=self._path,
    )

    new_dim_idx = 0
    for old_dim in remaining_dims:
        if self._unit_converters[old_dim] is not None:
            proj_spectrum._unit_converters[new_dim_idx] = self._unit_converters[old_dim]
        if self._ppm_scales[old_dim] is not None:
            proj_spectrum._ppm_scales[new_dim_idx] = self._ppm_scales[old_dim]
        new_dim_idx += 1

    return proj_spectrum


def get_projection(
    self,
    axis: int,
    method: Literal['sum', 'max', 'min', 'mean'] = 'sum',
) -> np.ndarray:
    if axis < 0 or axis >= self._ndim:
        raise InvalidDimensionIndexError(axis, self._ndim)

    cache_key = (axis, method)
    if cache_key not in self._projections:
        if method == 'sum':
            self._projections[cache_key] = np.sum(self._data, axis=axis)
        elif method == 'max':
            self._projections[cache_key] = np.max(self._data, axis=axis)
        elif method == 'min':
            self._projections[cache_key] = np.min(self._data, axis=axis)
        elif method == 'mean':
            self._projections[cache_key] = np.mean(self._data, axis=axis)
        else:
            raise SpectrumProcessingError(
                f"Unknown projection method: {method}. "
                f"Use 'sum', 'max', 'min', or 'mean'."
            )

    return self._projections[cache_key]


def projection_f1(self) -> np.ndarray:
    if self._ndim != 2:
        raise InvalidDimensionalityError(self._ndim, expected=2)
    return get_projection(self, 0)


def projection_f2(self) -> np.ndarray:
    if self._ndim != 2:
        raise InvalidDimensionalityError(self._ndim, expected=2)
    return get_projection(self, 1)
