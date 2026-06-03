from __future__ import annotations

import os
import re
from pathlib import Path

import nmrglue as ng
import numpy as np
from nmrglue.fileio.fileiobase import unit_conversion

from echofox.core.time import TimeRange, TimeValue
from echofox.core.time.exceptions import InvalidTimeValueError
from echofox.core.typing import Number
from echofox.nmr.config import config

from .exceptions import (
    CouldNotReadSpectrumError,
    DimensionMismatchError,
    InvalidSpectrumFormatError,
    SpectrumFileNotFoundError,
)
from .types import SpectrumFormat


def from_file(
    cls,
    path: str,
    spectrum_format: SpectrumFormat = None,
    name: str | None = None,
):
    """
    Create NmrSpectrum from a file.
    """
    if spectrum_format is None:
        spectrum_format = config.default_spectrum_format

    if spectrum_format not in cls.SUPPORTED_FORMATS:
        raise InvalidSpectrumFormatError(spectrum_format, cls.SUPPORTED_FORMATS)

    if spectrum_format == "bruker":
        return _from_bruker(cls, path, name)
    if spectrum_format == "pipe":
        return _from_pipe(cls, path, name)
    raise InvalidSpectrumFormatError(spectrum_format, cls.SUPPORTED_FORMATS)


def _from_bruker(cls, path: str, name: str | None = None):
    """Create spectrum from Bruker format."""
    if not os.path.isdir(path):
        raise SpectrumFileNotFoundError(path, "directory")

    try:
        dic, data = ng.bruker.read_pdata(path)
        udic = ng.bruker.guess_udic(dic, data)
    except Exception as exc:
        raise CouldNotReadSpectrumError(path, str(exc))

    return _from_udic(cls, data, dic, udic, name, path, "bruker")


def _from_pipe(cls, path: str, name: str | None = None):
    """Create spectrum from NMRPipe format."""
    if not os.path.isfile(path):
        raise SpectrumFileNotFoundError(path, "file")

    try:
        dic, data = ng.pipe.read(path)
        udic = ng.pipe.guess_udic(dic, data)
    except Exception as exc:
        raise CouldNotReadSpectrumError(path, str(exc))

    return _from_udic(cls, data, dic, udic, name, path, "pipe")


def from_pseudo_nd(
    cls,
    path: str | list[str],
    spectrum_format: SpectrumFormat = "pipe",
    name: str | None = None,
    file_pattern: str = "*.ft*",
    pseudo_axis_values: list[Number | str] | None = None,
    pseudo_axis_label: str = "pseudo",
    pseudo_axis_unit: str | None = None,
):
    """
    Load pseudo-(N+1)D spectrum from multiple ND spectra.
    """
    if isinstance(path, str):
        if os.path.isdir(path):
            folder = Path(path)
            files = sorted(folder.glob(file_pattern), key=lambda p: p.name.lower())
        elif os.path.isfile(path):
            files = [Path(path)]
        else:
            raise SpectrumFileNotFoundError(path, "file or directory")
    else:
        files = []
        for p in path:
            if os.path.isfile(p):
                files.append(Path(p))
            else:
                raise SpectrumFileNotFoundError(p, "file")

    if not files:
        raise SpectrumFileNotFoundError(
            str(path),
            f"files matching pattern '{file_pattern}'",
        )

    spectra = []
    for f in files:
        try:
            spec = cls.from_file(str(f), spectrum_format)
            spectra.append(spec)
        except Exception as exc:
            raise CouldNotReadSpectrumError(str(f), str(exc))

    if not spectra:
        raise SpectrumFileNotFoundError(str(path), "readable spectra")

    base_shape = spectra[0].shape
    base_ndim = spectra[0].ndim
    for i, spec in enumerate(spectra[1:], 1):
        if spec.shape != base_shape:
            raise DimensionMismatchError(f"spectrum[{i}] shape", base_shape, spec.shape)

    if pseudo_axis_values is not None:
        if len(pseudo_axis_values) != len(spectra):
            raise DimensionMismatchError(
                "pseudo_axis_values", len(spectra), len(pseudo_axis_values)
            )

    stacked_data = np.stack([s.data for s in spectra], axis=0)

    def _coerce_to_float(values: list[Number | str]) -> list[float] | None:
        numeric_values: list[float] = []
        for value in values:
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                return None
        return numeric_values

    def _coerce_to_seconds(values: list[Number | str]) -> list[float] | None:
        seconds_values: list[float] = []
        for value in values:
            try:
                if pseudo_axis_unit and isinstance(value, (int, float)):
                    seconds_values.append(
                        TimeValue(f"{value} {pseudo_axis_unit}").seconds
                    )
                elif (
                    pseudo_axis_unit
                    and isinstance(value, str)
                    and re.search(r"[a-zA-Z]", value) is None
                ):
                    seconds_values.append(
                        TimeValue(f"{value} {pseudo_axis_unit}").seconds
                    )
                else:
                    seconds_values.append(TimeValue(value).seconds)
            except InvalidTimeValueError:
                # Non-time labels (e.g., "saturated"/"unsaturated") are valid for some pseudo axes.
                return None
            except Exception:
                return None
        return seconds_values

    pseudo_axis_numeric: list[float] | None = None
    pseudo_axis_seconds: list[float] | None = None

    if pseudo_axis_values is not None:
        label_is_time = bool(pseudo_axis_label) and "time" in pseudo_axis_label.lower()
        values_have_units = any(
            isinstance(value, str) and re.search(r"[a-zA-Z]", value)
            for value in pseudo_axis_values
        )
        if pseudo_axis_unit or label_is_time or values_have_units:
            pseudo_axis_seconds = _coerce_to_seconds(pseudo_axis_values)
        if pseudo_axis_seconds is not None:
            pseudo_range = TimeRange(min(pseudo_axis_seconds), max(pseudo_axis_seconds))
            pseudo_axis_scale = np.array(pseudo_axis_seconds, dtype=float)
        else:
            pseudo_axis_numeric = _coerce_to_float(pseudo_axis_values)
            if pseudo_axis_numeric is not None:
                pseudo_range = (min(pseudo_axis_numeric), max(pseudo_axis_numeric))
                pseudo_axis_scale = np.array(pseudo_axis_numeric, dtype=float)
            else:
                pseudo_range = (0.0, float(len(spectra) - 1))
                pseudo_axis_scale = np.arange(len(spectra), dtype=float)
    else:
        pseudo_range = (0.0, float(len(spectra) - 1))
        pseudo_axis_scale = np.arange(len(spectra), dtype=float)

    dimension_ranges = [pseudo_range] + spectra[0].dimension_ranges

    base_nuclei = spectra[0].nuclei
    if base_nuclei:
        nuclei = [pseudo_axis_label] + base_nuclei
    else:
        nuclei = [pseudo_axis_label] + ["unknown"] * base_ndim

    base_freqs = spectra[0].frequencies
    if base_freqs:
        frequencies = [None] + base_freqs
    else:
        frequencies = None

    result = cls(
        data=stacked_data,
        dimension_ranges=dimension_ranges,
        nuclei=nuclei,
        frequencies=frequencies,
        name=name,
        path=str(path) if isinstance(path, str) else None,
        pseudo_axis_values=pseudo_axis_values,
        pseudo_axis_label=pseudo_axis_label,
        pseudo_axis_unit=pseudo_axis_unit,
        source_files=[str(f) for f in files],
    )

    for dim in range(base_ndim):
        if spectra[0]._unit_converters[dim] is not None:
            result._unit_converters[dim + 1] = spectra[0]._unit_converters[dim]

    for dim in range(base_ndim):
        result._ppm_scales[dim + 1] = spectra[0].get_ppm_axis(dim)

    result._ppm_scales[0] = pseudo_axis_scale

    return result


def _from_udic(
    cls,
    data: np.ndarray,
    dic: dict,
    udic: dict,
    name: str | None,
    path: str,
    spectrum_format: str,
):
    """Create spectrum from universal dictionary."""
    ndim = data.ndim

    dimension_ranges = []
    nuclei = []
    frequencies = []
    unit_converters = []

    for dim in range(ndim):
        if spectrum_format == "bruker":
            if ndim == 1:
                procs_key = "procs"
            else:
                procs_key = f"proc{ndim - dim}s" if dim < ndim - 1 else "procs"
                if procs_key not in dic:
                    procs_key = "procs" if dim == ndim - 1 else f"proc{ndim - dim}s"

            if procs_key in dic:
                uc = unit_conversion(
                    data.shape[dim],
                    True,
                    dic[procs_key]["SW_p"],
                    dic[procs_key]["SF"],
                    (
                        dic[procs_key]["OFFSET"]
                        - (dic[procs_key]["SW_p"] / dic[procs_key]["SF"]) / 2
                    )
                    * dic[procs_key]["SF"],
                )
            else:
                uc = unit_conversion(
                    data.shape[dim],
                    True,
                    udic[dim]["sw"],
                    udic[dim]["obs"],
                    udic[dim]["car"],
                )
        else:
            uc = ng.pipe.make_uc(dic, data, dim=dim)

        unit_converters.append(uc)

        ppm_max, ppm_min = uc.ppm_limits()
        dimension_ranges.append((ppm_min, ppm_max))

        label = udic[dim]["label"]
        if label == "HN":
            label = "1H"
        nuclei.append(label)

        frequencies.append(udic[dim]["obs"])

    spectrum = cls(
        data=data,
        dimension_ranges=dimension_ranges,
        nuclei=nuclei,
        frequencies=frequencies,
        name=name,
        path=path,
        spectrum_format=spectrum_format,
        dic=dic,
        udic=udic,
    )

    spectrum._unit_converters = unit_converters
    return spectrum
