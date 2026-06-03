from __future__ import annotations

from typing import Literal, Tuple, Union

import nmrglue as ng
import numpy as np

from echofox.nmr.chemical_shift import ChemicalShift, PpmRange
from echofox.nmr.peak import NmrPeak, PeakList
from .exceptions import InvalidDimensionalityError
from .models import _PeakCandidate


def prepare_pick_2d_parameters(
    self,
    sino: int | float | None,
    ppm_range: Tuple[
        Union[PpmRange, Tuple[float | str | ChemicalShift, float | str | ChemicalShift]],
        Union[PpmRange, Tuple[float | str | ChemicalShift, float | str | ChemicalShift]],
    ],
) -> tuple[float, list[PpmRange]]:
    if self.ndim != 2:
        raise InvalidDimensionalityError(
            f"pick_2d() requires a 2D spectrum, got {self.ndim}D. "
            f"Use pick_1d() for 1D or pick_pseudo_nd() for pseudo-nD spectra."
        )

    if self.is_pseudo_nd:
        raise InvalidDimensionalityError(
            f"pick_2d() cannot be used on pseudo-nD spectra. "
            f"Use pick_pseudo_nd() to pick peaks from all planes."
        )

    if sino is None:
        sino = self.min_sino

    ppm_filters: list[PpmRange] = []
    if ppm_range != (None, None):
        for current_range, default_range in zip(ppm_range, self._dimension_ranges[:2]):
            if current_range is None:
                ppm_filters.append(default_range)
            elif isinstance(current_range, PpmRange):
                ppm_filters.append(current_range)
            else:
                ppm_filters.append(PpmRange(*current_range))
    else:
        ppm_filters = [self._dimension_ranges[0], self._dimension_ranges[1]]

    return float(sino), ppm_filters


def fit_gaussian_multipoint(
    intensities: np.ndarray,
    center_idx: int,
    max_points: int = 3,
    consistency_threshold: float = 0.1,
) -> tuple[float, float, float, float, int, float]:
    n_total = len(intensities)
    y_center = float(intensities[center_idx])

    if center_idx == 0 or center_idx == n_total - 1:
        return 0.0, y_center, 0.0, 0.0, 1, 0.0

    if max_points % 2 == 0:
        max_points += 1

    best_result = None
    prev_params = None

    for n_points in range(3, max_points + 1, 2):
        half_width = n_points // 2

        if center_idx - half_width < 0 or center_idx + half_width >= n_total:
            break

        indices = np.arange(center_idx - half_width, center_idx + half_width + 1)
        y_values = intensities[indices].astype(float)

        if np.any(y_values <= 0):
            break

        if y_center < np.max(y_values[y_values != y_center]):
            break

        left_side = y_values[:half_width]
        right_side = y_values[half_width + 1 :]

        if not np.all(np.diff(left_side) > 0):
            break

        if not np.all(np.diff(right_side) < 0):
            break

        x_values = np.arange(-half_width, half_width + 1, dtype=float)
        ln_y = np.log(y_values)

        A = np.column_stack([x_values**2, x_values, np.ones_like(x_values)])

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(A, ln_y, rcond=None)
            a, b, c = coeffs

            if a >= 0:
                break

            offset = -b / (2 * a)

            if abs(offset) > 0.5:
                offset = np.clip(offset, -0.5, 0.5)

            ln_intensity_max = c - (b**2) / (4 * a)
            fitted_intensity = np.exp(ln_intensity_max)

            sigma_squared = -1.0 / (2.0 * a)
            if sigma_squared <= 0:
                break

            sigma = np.sqrt(sigma_squared)
            fwhm = 2.35482 * sigma

            if len(residuals) > 0:
                ss_res = residuals[0] if residuals.size > 0 else 0.0
                ss_tot = np.sum((ln_y - np.mean(ln_y)) ** 2)
                fit_quality = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                fit_quality = max(0.0, min(1.0, fit_quality))
            else:
                fit_quality = 1.0

            current_params = (offset, fitted_intensity, fwhm)

            if prev_params is not None:
                rel_offset_change = abs(current_params[0] - prev_params[0]) / (
                    abs(prev_params[0]) + 1e-6
                )
                rel_intensity_change = abs(current_params[1] - prev_params[1]) / (
                    prev_params[1] + 1e-6
                )
                rel_fwhm_change = abs(current_params[2] - prev_params[2]) / (
                    prev_params[2] + 1e-6
                )

                max_change = max(rel_offset_change, rel_intensity_change, rel_fwhm_change)

                if max_change > consistency_threshold:
                    break

            best_result = (
                float(offset),
                float(fitted_intensity),
                float(fwhm),
                float(sigma),
                n_points,
                float(fit_quality),
            )
            prev_params = current_params

        except (np.linalg.LinAlgError, ValueError):
            break

    if best_result is not None:
        return best_result
    return 0.0, y_center, 0.0, 0.0, 1, 0.0


def fit_lorentzian_multipoint(
    intensities: np.ndarray,
    center_idx: int,
    max_points: int = 3,
    consistency_threshold: float = 0.1,
) -> tuple[float, float, float, float, int, float]:
    n_total = len(intensities)
    y_center = float(intensities[center_idx])

    if center_idx == 0 or center_idx == n_total - 1:
        return 0.0, y_center, 0.0, 0.0, 1, 0.0

    if max_points % 2 == 0:
        max_points += 1

    best_result = None
    prev_params = None

    for n_points in range(3, max_points + 1, 2):
        half_width = n_points // 2
        if center_idx - half_width < 0 or center_idx + half_width >= n_total:
            break

        indices = np.arange(center_idx - half_width, center_idx + half_width + 1)
        y_values = intensities[indices].astype(float)

        if np.any(y_values <= 0):
            break
        if y_center < np.max(y_values[y_values != y_center]):
            break

        left_side = y_values[:half_width]
        right_side = y_values[half_width + 1 :]
        if not np.all(np.diff(left_side) > 0):
            break
        if not np.all(np.diff(right_side) < 0):
            break

        x_values = np.arange(-half_width, half_width + 1, dtype=float)
        inv_y = 1.0 / y_values
        A = np.column_stack([x_values**2, x_values, np.ones_like(x_values)])

        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(A, inv_y, rcond=None)
            a, b, c = coeffs

            if a <= 0:
                break

            offset = -b / (2 * a)
            if abs(offset) > 0.5:
                offset = np.clip(offset, -0.5, 0.5)

            q = c - a * (offset**2)
            if q <= 0:
                break

            fitted_intensity = 1.0 / q
            gamma_sq = q / a
            if gamma_sq <= 0:
                break
            gamma = np.sqrt(gamma_sq)
            fwhm = 2.0 * gamma

            if len(residuals) > 0:
                ss_res = residuals[0] if residuals.size > 0 else 0.0
                ss_tot = np.sum((inv_y - np.mean(inv_y)) ** 2)
                fit_quality = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                fit_quality = max(0.0, min(1.0, fit_quality))
            else:
                fit_quality = 1.0

            current_params = (offset, fitted_intensity, fwhm)
            if prev_params is not None:
                rel_offset_change = abs(current_params[0] - prev_params[0]) / (
                    abs(prev_params[0]) + 1e-6
                )
                rel_intensity_change = abs(current_params[1] - prev_params[1]) / (
                    prev_params[1] + 1e-6
                )
                rel_fwhm_change = abs(current_params[2] - prev_params[2]) / (
                    prev_params[2] + 1e-6
                )
                if (
                    max(rel_offset_change, rel_intensity_change, rel_fwhm_change)
                    > consistency_threshold
                ):
                    break

            best_result = (
                float(offset),
                float(fitted_intensity),
                float(fwhm),
                float(gamma),
                n_points,
                float(fit_quality),
            )
            prev_params = current_params

        except (np.linalg.LinAlgError, ValueError):
            break

    if best_result is not None:
        return best_result
    return 0.0, y_center, 0.0, 0.0, 1, 0.0


def build_peaklist_from_candidates(
    self,
    candidates: list[_PeakCandidate],
    ppm_filters: list[PpmRange],
    fit_method: Literal["gaussian", "lorentzian"] = "gaussian",
) -> PeakList:
    data_shape_f1, data_shape_f2 = self.data.shape

    resolution_f1_hz = (
        self.dimension_ranges[0].width * self.frequencies[0] / self.shape[0]
    )
    resolution_f2_hz = (
        self.dimension_ranges[1].width * self.frequencies[1] / self.shape[1]
    )

    nuclei_f1 = self.nuclei[0] if self.nuclei else None
    nuclei_f2 = self.nuclei[1] if self.nuclei else None
    freq_f1 = self.frequencies[0] if self.frequencies else None
    freq_f2 = self.frequencies[1] if self.frequencies else None

    peaklist = PeakList()

    fit_method = (fit_method or "gaussian").lower()
    if fit_method not in ("gaussian", "lorentzian"):
        raise ValueError(f"Unsupported fit_method '{fit_method}'")
    fit_function = (
        fit_gaussian_multipoint if fit_method == "gaussian" else fit_lorentzian_multipoint
    )

    for candidate in candidates:
        center_f1_idx = int(candidate.center[0])
        center_f2_idx = int(candidate.center[1])

        if not (
            0 <= center_f1_idx < data_shape_f1 and 0 <= center_f2_idx < data_shape_f2
        ):
            continue

        adjusted_f1_idx = float(center_f1_idx)
        adjusted_f2_idx = float(center_f2_idx)

        center_intensity = float(self.data[center_f1_idx, center_f2_idx])

        f1_fitted_intensity = center_intensity
        f2_fitted_intensity = center_intensity
        f1_fwhm_pts = 0.0
        f2_fwhm_pts = 0.0
        f1_shape_param_pts = 0.0
        f2_shape_param_pts = 0.0
        f1_n_points = 1
        f2_n_points = 1
        f1_quality = 0.0
        f2_quality = 0.0

        f1_slice = self.data[:, center_f2_idx]
        (
            f1_offset,
            f1_fitted_intensity,
            f1_fwhm_pts,
            f1_shape_param_pts,
            f1_n_points,
            f1_quality,
        ) = fit_function(f1_slice, center_f1_idx, max_points=7, consistency_threshold=0.1)
        adjusted_f1_idx += f1_offset

        f2_slice = self.data[center_f1_idx, :]
        (
            f2_offset,
            f2_fitted_intensity,
            f2_fwhm_pts,
            f2_shape_param_pts,
            f2_n_points,
            f2_quality,
        ) = fit_function(f2_slice, center_f2_idx, max_points=7, consistency_threshold=0.1)
        adjusted_f2_idx += f2_offset

        f1_shift = self._unit_converters[0].ppm(adjusted_f1_idx)
        f2_shift = self._unit_converters[1].ppm(adjusted_f2_idx)

        intensity = (f1_fitted_intensity + f2_fitted_intensity) / 2.0

        fitted_linewidths = None
        gaussian_sigmas = None
        lorentzian_gammas = None
        volume_estimate = None
        if f1_fwhm_pts > 0 and f2_fwhm_pts > 0:
            f1_lw_hz = f1_fwhm_pts * resolution_f1_hz
            f2_lw_hz = f2_fwhm_pts * resolution_f2_hz
            fitted_linewidths = [f1_lw_hz, f2_lw_hz]
        if fit_method == "gaussian" and f1_shape_param_pts > 0 and f2_shape_param_pts > 0:
            gaussian_sigmas = [
                f1_shape_param_pts * resolution_f1_hz,
                f2_shape_param_pts * resolution_f2_hz,
            ]
            volume_estimate = intensity * (
                2.0 * np.pi * f1_shape_param_pts * f2_shape_param_pts
            )
        elif (
            fit_method == "lorentzian"
            and f1_shape_param_pts > 0
            and f2_shape_param_pts > 0
        ):
            lorentzian_gammas = [
                f1_shape_param_pts * resolution_f1_hz,
                f2_shape_param_pts * resolution_f2_hz,
            ]
            volume_estimate = (
                intensity
                * (np.pi * f1_shape_param_pts / 2)
                * (np.pi * f2_shape_param_pts / 2)
            )
        elif candidate.linewidth_pts is not None:
            f1_pts, f2_pts = candidate.linewidth_pts
            fitted_linewidths = [
                float(f1_pts) * resolution_f1_hz,
                float(f2_pts) * resolution_f2_hz,
            ]

        peak_kwargs = dict(
            nuclei=[nuclei_f1, nuclei_f2],
            intensity=intensity,
            frequencies=[freq_f1, freq_f2],
        )
        if fitted_linewidths is not None:
            peak_kwargs["linewidths"] = fitted_linewidths
        if gaussian_sigmas is not None:
            peak_kwargs["gaussian_sigmas"] = gaussian_sigmas
        if lorentzian_gammas is not None:
            peak_kwargs["lorentzian_gammas"] = lorentzian_gammas
        if volume_estimate is not None:
            peak_kwargs["volume"] = volume_estimate
        elif candidate.volume is not None:
            peak_kwargs["volume"] = candidate.volume

        try:
            adjusted_peak = NmrPeak(
                [f1_shift, f2_shift],
                **peak_kwargs,
            )

            adjusted_peak._extra_properties["fit_f1_n_points"] = f1_n_points
            adjusted_peak._extra_properties["fit_f2_n_points"] = f2_n_points
            adjusted_peak._extra_properties["fit_f1_quality"] = f1_quality
            adjusted_peak._extra_properties["fit_f2_quality"] = f2_quality
            adjusted_peak._extra_properties["fit_quality_avg"] = (
                f1_quality + f2_quality
            ) / 2.0

        except Exception as exc:
            log.exception("Exception in NmrSpectrum")
            import warnings

            warnings.warn(
                f"Failed to create adjusted peak: {exc}. Skipping peak.",
                stacklevel=2,
            )
            continue

        if not (
            ppm_filters[0].contains(adjusted_peak.get_chemical_shift(0))
            and ppm_filters[1].contains(adjusted_peak.get_chemical_shift(1))
        ):
            continue

        peaklist.add(adjusted_peak)

    return peaklist


def pick_2d(
    self,
    sino: int | float | None = None,
    msep=(1, 1),
    edge=0,
    ppm_range: Tuple[
        Union[PpmRange, Tuple[float | str | ChemicalShift, float | str | ChemicalShift]],
        Union[PpmRange, Tuple[float | str | ChemicalShift, float | str | ChemicalShift]],
    ] = (None, None),
    peaklist_name: str | None = None,
    fit_method: Literal["gaussian", "lorentzian"] = "gaussian",
) -> PeakList | None:
    sino, ppm_filters = prepare_pick_2d_parameters(self, sino, ppm_range)

    if edge > 0:
        edge_free_data = self.data[edge:-edge, edge:-edge]
    else:
        edge_free_data = self.data

    try:
        raw_peaks = ng.peakpick.pick(
            edge_free_data,
            pthres=self.noise_level * sino,
            msep=msep,
            algorithm="thres",
            lineshapes=["gauss", "gauss"],
        )
        peaks: list[_PeakCandidate] = []
        for peak in raw_peaks:
            center = (float(peak[0]) + edge, float(peak[1]) + edge)
            linewidth_pts = (float(peak[3]), float(peak[4])) if len(peak) > 4 else None
            volume = float(peak[5]) if len(peak) > 5 else None
            peaks.append(
                _PeakCandidate(
                    center=center,
                    linewidth_pts=linewidth_pts,
                    volume=volume,
                )
            )

    except (AttributeError, ValueError, IndexError) as exc:
        self._log.warning("Peak picking failed: %s", exc)
        peaks = []

    peaklist = build_peaklist_from_candidates(
        self,
        peaks,
        ppm_filters,
        fit_method=fit_method,
    )

    if peaklist_name is not None:
        peaklist.name = peaklist_name
        self.peaklists[peaklist_name] = peaklist
    else:
        peaklist.name = self.DEFAULT_PEAKLIST_KEY
        self.peaklists.add(self.DEFAULT_PEAKLIST_KEY, peaklist)

    return peaklist


def pick_pseudo_nd(
    self,
    sino: int | float | None = None,
    msep=(1, 1),
    edge=0,
    ppm_range: Tuple[
        Union[PpmRange, Tuple[float | str | ChemicalShift, float | str | ChemicalShift]],
        Union[PpmRange, Tuple[float | str | ChemicalShift, float | str | ChemicalShift]],
    ] = (None, None),
    time_param: str = "delay_time",
    fit_method: Literal["gaussian", "lorentzian"] = "gaussian",
) -> PeakList:
    if not self.is_pseudo_nd:
        raise ValueError(
            "Spectrum must be pseudo-nD. Use pick_2d() for regular 2D spectra."
        )

    if self.ndim != 3:
        raise ValueError(
            f"pick_pseudo_nd currently only supports pseudo-3D (got {self.ndim}D). "
            "Use pick_2d() for 2D or implement picking for higher dimensions."
        )

    if self.pseudo_axis_values is None:
        raise ValueError("Pseudo-nD spectrum must have pseudo_axis_values defined")

    n_planes = self.shape[0]

    if len(self.pseudo_axis_values) != n_planes:
        raise ValueError(
            f"Mismatch: {n_planes} planes but {len(self.pseudo_axis_values)} pseudo_axis_values"
        )

    combined_peaklist = PeakList()

    for plane_idx in range(n_planes):
        plane_data = self.data[plane_idx, :, :]

        temp_spectrum = self.__class__(
            data=plane_data,
            dimension_ranges=[self._dimension_ranges[1], self._dimension_ranges[2]],
            nuclei=[self.nuclei[1], self.nuclei[2]] if self.nuclei else None,
            frequencies=[self.frequencies[1], self.frequencies[2]]
            if self.frequencies
            else None,
            name=f"{self.name}_plane{plane_idx}" if self.name else None,
        )

        # We have to copy over the correct unit_conversion and ppm_scales.
        # recreation does not work --> leads to small shifts

        if self._unit_converters[1] is not None:
            temp_spectrum._unit_converters[0] = self._unit_converters[1]
        if self._unit_converters[2] is not None:
            temp_spectrum._unit_converters[1] = self._unit_converters[2]
        if self._ppm_scales[1] is not None:
            temp_spectrum._ppm_scales[0] = self._ppm_scales[1]
        if self._ppm_scales[2] is not None:
            temp_spectrum._ppm_scales[1] = self._ppm_scales[2]

        plane_peaklist = pick_2d(
            temp_spectrum,
            sino=sino,
            msep=msep,
            edge=edge,
            ppm_range=ppm_range,
            fit_method=fit_method,
        )

        pseudo_value = self.pseudo_axis_values[plane_idx]
        for peak in plane_peaklist:
            peak._extra_properties[time_param] = pseudo_value
            peak._extra_properties["plane_index"] = plane_idx

        combined_peaklist.extend(plane_peaklist)

    self.peaklist = combined_peaklist

    return self.peaklist
