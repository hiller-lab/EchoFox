"""
NMR Spectrum representation for n-dimensional spectral data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from nmrglue.fileio.fileiobase import unit_conversion

from echofox.core.time import TimeRange, TimeValue
from echofox.core.typing import Number
from echofox.nmr.chemical_shift import ChemicalShift, PpmRange
from echofox.nmr.config import config
from echofox.nmr.peak import NmrPeak, PeakList, PeakListCollection

from . import axes as spectrum_axes
from . import intensity as spectrum_intensity
from . import io as spectrum_io
from . import noise as spectrum_noise
from . import peak_picking as spectrum_peak_picking
from . import projections as spectrum_projections
from . import serialization as spectrum_serialization
from . import slicing as spectrum_slicing
from .exceptions import DimensionMismatchError, InvalidSpectrumDataError
from .models import _PeakCandidate
from .types import SpectrumFormat


class NmrSpectrum:
    """
    Represents an n-dimensional NMR spectrum.

    The NmrSpectrum class stores spectral data along with associated metadata
    including chemical shift ranges, nucleus types, and acquisition parameters.
    It supports reading from various file formats (Bruker, NMRPipe, CSV).

    Examples:
        # Create from file
        spectrum = NmrSpectrum.from_file(
            path="/path/to/spectrum",
            spectrum_format='bruker'
        )

        # Create from raw data
        spectrum = NmrSpectrum(
            data=np.array([...]),
            dimension_ranges=[(0.0, 10.0)],
            nuclei=['1H'],
            frequencies=[600.13]
        )

        # 2D spectrum (e.g., HSQC)
        spectrum_2d = NmrSpectrum(
            data=np.array([[...]]),
            dimension_ranges=[(0.0, 10.0), (100.0, 140.0)],
            nuclei=['1H', '15N'],
            frequencies=[600.13, 60.82]
        )

    Args:
        data: Numpy array containing spectral intensities
        dimension_ranges: List of PpmRange/TimeRange objects or (low, high) tuples for each dimension
        nuclei: List of nucleus labels for each dimension (optional)
        frequencies: List of spectrometer frequencies in MHz for each dimension (optional)
        acquisition_date: Timestamp of acquisition (optional)
        processing_info: Dictionary of processing parameters (optional)
        spectrum_format: Spectrum format identifier (optional)
        name: Name identifier for the spectrum (optional)
        path: File path if loaded from file (optional)
        pseudo_axis_values: Values (numeric or labels) for pseudo dimension if stacked from multiple spectra (optional)
        pseudo_axis_label: Label for pseudo dimension (e.g., 'time', 'concentration') (optional)
        pseudo_axis_unit: Unit for pseudo dimension values (e.g., 'ms', 's') (optional)
        source_files: List of source file paths if loaded from pseudo-ND (optional)
        dic: Raw nmrglue dictionary from file (optional)
        udic: Universal dictionary from nmrglue (optional)
        peaklist: List of Peak objects (optional, deprecated in favor of peaklists)
        peaklists: PeakListCollection (optional)
        **kwargs: Additional metadata
    """

    SUPPORTED_FORMATS = ["bruker", "pipe"]
    DEFAULT_PEAKLIST_KEY = "default"

    def __init__(
        self,
        data: np.ndarray,
        dimension_ranges: list[PpmRange | TimeRange | tuple[float, float]],
        nuclei: list[str] | None = None,
        frequencies: list[float] | None = None,
        acquisition_date: datetime | None = None,
        processing_info: dict | None = None,
        spectrum_format: str | None = None,
        name: str | None = None,
        path: str | list[str] | None = None,
        pseudo_axis_values: list[Number | str] | None = None,
        pseudo_axis_label: str | None = None,
        pseudo_axis_unit: str | None = None,
        source_files: list[str] | None = None,
        dic: dict | None = None,
        udic: dict | None = None,
        peaklist: list[NmrPeak] | None = None,
        peaklists: PeakListCollection | None = None,
        **kwargs,
    ):
        if not isinstance(data, np.ndarray):
            raise InvalidSpectrumDataError(f"Data must be numpy array, got {type(data).__name__}")

        self._data = data
        self._ndim = data.ndim
        self._name = name
        self._path = path
        self._spectrum_format = spectrum_format
        # Peak list collection
        if peaklists is not None:
            self._peaklists = peaklists
        else:
            self._peaklists = PeakListCollection(name=f"{name}_peaklists" if name else None)
        if peaklist is not None:
            self.peaklist = peaklist

        # Validate and convert dimension_ranges to range objects
        if len(dimension_ranges) != self._ndim:
            raise DimensionMismatchError("dimension_ranges", self._ndim, len(dimension_ranges))

        self._dimension_ranges: list[PpmRange | TimeRange] = []
        for r in dimension_ranges:
            if isinstance(r, (PpmRange, TimeRange)):
                self._dimension_ranges.append(r)
            else:
                # Convert tuple to PpmRange (assumes spectral ppm)
                self._dimension_ranges.append(PpmRange(r[0], r[1], validate_range=False))

        # Validate and set nuclei
        if nuclei is not None:
            if len(nuclei) != self._ndim:
                raise DimensionMismatchError("nuclei", self._ndim, len(nuclei))
            self._nuclei = [str(n) for n in nuclei]
        else:
            self._nuclei = None

        # Validate and set frequencies
        if frequencies is not None:
            if len(frequencies) != self._ndim:
                raise DimensionMismatchError("frequencies", self._ndim, len(frequencies))
            self._frequencies = [float(f) if f is not None else None for f in frequencies]
        else:
            self._frequencies = None

        self._acquisition_date = acquisition_date
        self._processing_info = processing_info or {}
        self._metadata = kwargs

        # Pseudo-ND specific attributes
        self._pseudo_axis_values = pseudo_axis_values
        self._pseudo_axis_label = pseudo_axis_label
        self._pseudo_axis_unit = pseudo_axis_unit
        self._source_files = source_files

        # Raw file dictionaries (from nmrglue)
        self._dic = dic
        self._udic = udic

        # Unit converters for each dimension
        self._unit_converters: list[unit_conversion | None] = [None] * self._ndim

        # Cached properties
        self._noise_level: float | None = None
        self._min_sino: float = config.default_min_sino
        self._ppm_scales: list[np.ndarray | None] = [None] * self._ndim
        self._projections: dict = {}  # Cache keyed by (axis, method)

        # Initialize unit converters if we have frequencies
        self._init_unit_converters()

    # -------------------------------------------------------------------------
    # Class methods for creating spectra from files
    # -------------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str, spectrum_format: SpectrumFormat = None, name: str | None = None) -> NmrSpectrum:
        return spectrum_io.from_file(cls, path, spectrum_format, name)

    @classmethod
    def _from_bruker(cls, path: str, name: str | None = None) -> NmrSpectrum:
        return spectrum_io._from_bruker(cls, path, name)

    @classmethod
    def _from_pipe(cls, path: str, name: str | None = None) -> NmrSpectrum:
        return spectrum_io._from_pipe(cls, path, name)

    @classmethod
    def from_pseudo_nd(
        cls,
        path: str | list[str],
        spectrum_format: SpectrumFormat = "pipe",
        name: str | None = None,
        file_pattern: str = "*.ft*",
        pseudo_axis_values: list[Number | str] | None = None,
        pseudo_axis_label: str = "pseudo",
        pseudo_axis_unit: str | None = None,
    ) -> NmrSpectrum:
        return spectrum_io.from_pseudo_nd(
            cls,
            path,
            spectrum_format=spectrum_format,
            name=name,
            file_pattern=file_pattern,
            pseudo_axis_values=pseudo_axis_values,
            pseudo_axis_label=pseudo_axis_label,
            pseudo_axis_unit=pseudo_axis_unit,
        )

    @classmethod
    def _from_udic(
        cls,
        data: np.ndarray,
        dic: dict,
        udic: dict,
        name: str | None,
        path: str,
        spectrum_format: str,
    ) -> NmrSpectrum:
        return spectrum_io._from_udic(cls, data, dic, udic, name, path, spectrum_format)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """Returns spectral data array."""
        return self._data

    @property
    def peaklists(self) -> PeakListCollection:
        """Returns the PeakListCollection associated with the spectrum."""
        return self._peaklists

    @peaklists.setter
    def peaklists(self, value: PeakListCollection) -> None:
        if not isinstance(value, PeakListCollection):
            raise TypeError(f"Expected PeakListCollection, got {type(value)}")
        self._peaklists = value

    @property
    def peaklist(self) -> PeakList | None:
        """Convenience accessor for the default peak list (for backward compatibility)."""
        return self._peaklists.get(self.DEFAULT_PEAKLIST_KEY)

    @peaklist.setter
    def peaklist(self, value: list[NmrPeak] | PeakList | None) -> None:
        if value is None:
            if self.DEFAULT_PEAKLIST_KEY in self._peaklists:
                del self._peaklists[self.DEFAULT_PEAKLIST_KEY]
            return
        if isinstance(value, PeakList):
            peaklist = value
        else:
            peaklist = PeakList()
            for peak in value:
                peaklist.add(peak)
        self._peaklists[self.DEFAULT_PEAKLIST_KEY] = peaklist

    def load_peaklists(self, file_path: str | Path) -> PeakListCollection:
        """
        Load a PeakListCollection from a file and attach it to this spectrum.

        Args:
            file_path: JSON file created via PeakListCollection.save

        Returns:
            The loaded PeakListCollection
        """
        self._peaklists = PeakListCollection.load(file_path)
        return self._peaklists

    def save_peaklists(self, file_path: str | Path, indent: int = 2) -> None:
        """
        Save the current PeakListCollection to a file.

        Args:
            file_path: Output path for JSON
            indent: JSON indentation for readability (default: 2)
        """
        self._peaklists.save(file_path, indent=indent)

    @property
    def ndim(self) -> int:
        """Returns number of dimensions."""
        return self._ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns shape of spectral data."""
        return self._data.shape

    @property
    def dimension_ranges(self) -> list[PpmRange | TimeRange]:
        """Returns list of range objects for each dimension."""
        return self._dimension_ranges.copy()

    @property
    def nuclei(self) -> list[str] | None:
        """Returns list of nucleus labels."""
        return self._nuclei.copy() if self._nuclei else None

    @property
    def frequencies(self) -> list[float] | None:
        """Returns list of spectrometer frequencies in MHz."""
        return self._frequencies.copy() if self._frequencies else None

    @property
    def acquisition_date(self) -> datetime | None:
        """Returns acquisition timestamp."""
        return self._acquisition_date

    @property
    def processing_info(self) -> dict:
        """Returns processing parameters."""
        return self._processing_info.copy()

    @property
    def metadata(self) -> dict:
        """Returns additional metadata."""
        return self._metadata.copy()

    @property
    def name(self) -> str | None:
        """Returns spectrum name."""
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        """Set spectrum name."""
        self._name = value

    @property
    def path(self) -> str | None:
        """Returns file path if loaded from file."""
        return self._path

    @property
    def noise_level(self) -> float:
        """Returns estimated noise level."""
        if self._noise_level is None:
            self._noise_level = self.estimate_noise_level()
        return self._noise_level

    @noise_level.setter
    def noise_level(self, value: float) -> None:
        """Set noise level."""
        self._noise_level = float(value)

    @property
    def min_sino(self) -> float:
        """Returns minimum signal-to-noise ratio for peak picking."""
        return self._min_sino

    @min_sino.setter
    def min_sino(self, value: float) -> None:
        """Set minimum signal-to-noise ratio."""
        self._min_sino = float(value)

    @property
    def is_pseudo_nd(self) -> bool:
        """Returns True if this spectrum was created from stacked spectra."""
        return self._pseudo_axis_values is not None or self._source_files is not None

    @property
    def pseudo_axis_values(self) -> list[Number | str] | None:
        """Returns the values for the pseudo dimension (e.g., time points)."""
        return self._pseudo_axis_values

    @property
    def pseudo_axis_label(self) -> str | None:
        """Returns the label for the pseudo dimension (e.g., 'time')."""
        return self._pseudo_axis_label

    @property
    def pseudo_axis_unit(self) -> str | None:
        """Returns the unit for the pseudo dimension (e.g., 'ms')."""
        return self._pseudo_axis_unit

    @property
    def source_files(self) -> list[str] | None:
        """Returns list of source files if loaded from pseudo-ND."""
        return self._source_files

    @property
    def dic(self) -> dict | None:
        """Returns raw nmrglue dictionary from file (Bruker/Pipe specific)."""
        return self._dic

    @property
    def udic(self) -> dict | None:
        """Returns universal dictionary from nmrglue."""
        return self._udic

    # -------------------------------------------------------------------------
    # Unit conversion and ppm scale methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _range_bounds(range_obj: PpmRange | TimeRange) -> tuple[float, float]:
        return spectrum_axes.range_bounds(range_obj)

    @staticmethod
    def _coerce_range_value(range_obj: PpmRange | TimeRange, value: float | int | str | TimeValue) -> float:
        return spectrum_axes.coerce_range_value(range_obj, value)

    def _init_unit_converters(self) -> None:
        spectrum_axes.init_unit_converters(self)

    def get_ppm_axis(self, dimension: int) -> np.ndarray:
        return spectrum_axes.get_ppm_axis(self, dimension)

    def get_ppm_scale(self, dimension: int) -> np.ndarray:
        """Alias for get_ppm_axis."""
        return spectrum_axes.get_ppm_scale(self, dimension)

    def ppm_to_index(self, dimension: int, ppm_value: float | int | str | TimeValue) -> int:
        return spectrum_axes.ppm_to_index(self, dimension, ppm_value)

    def index_to_ppm(self, dimension: int, index: int) -> float:
        return spectrum_axes.index_to_ppm(self, dimension, index)

    # -------------------------------------------------------------------------
    # Nucleus and frequency access
    # -------------------------------------------------------------------------

    def get_nucleus(self, dimension: int) -> str | None:
        return spectrum_axes.get_nucleus(self, dimension)

    def get_frequency(self, dimension: int) -> float | None:
        return spectrum_axes.get_frequency(self, dimension)

    def get_label_text(self, dimension: int) -> str:
        return spectrum_axes.get_label_text(self, dimension)

    # -------------------------------------------------------------------------
    # Extraction methods
    # -------------------------------------------------------------------------

    def get_row(self, idx: int | str | float | ChemicalShift) -> np.ndarray:
        return spectrum_slicing.get_row(self, idx)

    def get_column(self, idx: int | str | float | ChemicalShift) -> np.ndarray:
        return spectrum_slicing.get_column(self, idx)

    def get_segment(self, min_ppm: float, max_ppm: float, dimension: int = 0) -> np.ndarray:
        return spectrum_slicing.get_segment(self, min_ppm, max_ppm, dimension)

    def extract_segment(self, min_ppm: float, max_ppm: float, dimension: int = 0) -> NmrSpectrum:
        return spectrum_slicing.extract_segment(self, min_ppm, max_ppm, dimension)

    def extract_subspectrum(self, dimension_positions: dict, tolerance: float | None = None) -> NmrSpectrum:
        return spectrum_slicing.extract_subspectrum(self, dimension_positions, tolerance)

    def extract_trace(self, trace_dimension: int, dimension_positions: float | list[float] | dict) -> NmrSpectrum:
        return spectrum_slicing.extract_trace(self, trace_dimension, dimension_positions)

    def extract_projection(self, axis: int, method: Literal["sum", "max", "min", "mean"] = "sum") -> NmrSpectrum:
        return spectrum_projections.extract_projection(self, axis, method)

    # -------------------------------------------------------------------------
    # Projection methods
    # -------------------------------------------------------------------------

    def get_projection(self, axis: int, method: Literal["sum", "max", "min", "mean"] = "sum") -> np.ndarray:
        return spectrum_projections.get_projection(self, axis, method)

    @property
    def projection_f1(self) -> np.ndarray:
        """Get F1 projection (sum over rows) for 2D spectra."""
        return spectrum_projections.projection_f1(self)

    @property
    def projection_f2(self) -> np.ndarray:
        """Get F2 projection (sum over columns) for 2D spectra."""
        return spectrum_projections.projection_f2(self)

    def get_subspectrum(
        self, dimension_positions: dict, tolerance: float | None = None
    ) -> tuple[np.ndarray, dict, dict]:
        return spectrum_slicing.get_subspectrum(self, dimension_positions, tolerance)

    # -------------------------------------------------------------------------
    # Noise estimation and intensity methods
    # -------------------------------------------------------------------------

    def estimate_noise_level(self, method: str = None) -> float:
        return spectrum_noise.estimate_noise_level(self, method)

    def _median_absolute_deviation(self, k: float = 1.4826) -> float:
        return spectrum_noise.median_absolute_deviation(self, k)

    def get_max_intensity(self) -> float:
        """Get maximum intensity in spectrum."""
        return spectrum_intensity.get_max_intensity(self)

    def get_min_intensity(self) -> float:
        """Get minimum intensity in spectrum."""
        return spectrum_intensity.get_min_intensity(self)

    def get_intensity_at(self, *ppm_values: float) -> float:
        """
        Get intensity at specified ppm position.

        Args:
            *ppm_values: PPM values for each dimension

        Returns:
            Intensity value

        Raises:
            DimensionMismatchError: If wrong number of ppm values
        """
        return spectrum_intensity.get_intensity_at(self, *ppm_values)

    # -------------------------------------------------------------------------
    # Extent for plotting
    # -------------------------------------------------------------------------

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """
        Get extent for imshow plotting (2D spectra).

        Returns:
            (f2_max, f2_min, f1_max, f1_min)

        Raises:
            InvalidDimensionalityError: If spectrum is not 2D
        """
        return spectrum_intensity.extent(self)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Convert spectrum to dictionary representation.

        Note: Large spectral data is not included by default.

        Returns:
            Dictionary containing spectrum metadata
        """
        return spectrum_serialization.to_dict(self)

    # -------------------------------------------------------------------------
    # String representations
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return detailed representation."""
        return spectrum_serialization.spectrum_repr(self)

    def __str__(self) -> str:
        """Return string representation."""
        return spectrum_serialization.spectrum_str(self)

    # -------------------------------------------------------------------------
    # Peak picking work in progress
    # -------------------------------------------------------------------------

    def _prepare_pick_2d_parameters(
        self,
        sino: int | float | None,
        ppm_range: tuple[
            PpmRange | tuple[float | str | ChemicalShift, float | str | ChemicalShift],
            PpmRange | tuple[float | str | ChemicalShift, float | str | ChemicalShift],
        ],
    ) -> tuple[float, list[PpmRange]]:
        return spectrum_peak_picking.prepare_pick_2d_parameters(self, sino, ppm_range)

    @staticmethod
    def _fit_gaussian_multipoint(
        intensities: np.ndarray,
        center_idx: int,
        max_points: int = 5,
        consistency_threshold: float = 0.1,
    ) -> tuple[float, float, float, float, int, float]:
        return spectrum_peak_picking.fit_gaussian_multipoint(
            intensities,
            center_idx,
            max_points=max_points,
            consistency_threshold=consistency_threshold,
        )

    @staticmethod
    def _fit_lorentzian_multipoint(
        intensities: np.ndarray,
        center_idx: int,
        max_points: int = 5,
        consistency_threshold: float = 0.1,
    ) -> tuple[float, float, float, float, int, float]:
        return spectrum_peak_picking.fit_lorentzian_multipoint(
            intensities,
            center_idx,
            max_points=max_points,
            consistency_threshold=consistency_threshold,
        )

    def _build_peaklist_from_candidates(
        self,
        candidates: list[_PeakCandidate],
        ppm_filters: list[PpmRange],
        fit_method: Literal["gaussian", "lorentzian"] = "gaussian",
    ) -> PeakList:
        return spectrum_peak_picking.build_peaklist_from_candidates(
            self,
            candidates,
            ppm_filters,
            fit_method=fit_method,
        )

    def pick_2d(
        self,
        sino: int | float | None = None,
        msep=(1, 1),
        edge=0,
        ppm_range: tuple[
            PpmRange | tuple[float | str | ChemicalShift, float | str | ChemicalShift],
            PpmRange | tuple[float | str | ChemicalShift, float | str | ChemicalShift],
        ] = (None, None),
        peaklist_name: str | None = None,
        fit_method: Literal["gaussian", "lorentzian"] = "gaussian",
    ) -> PeakList | None:
        return spectrum_peak_picking.pick_2d(
            self,
            sino=sino,
            msep=msep,
            edge=edge,
            ppm_range=ppm_range,
            peaklist_name=peaklist_name,
            fit_method=fit_method,
        )

    def pick_pseudo_nd(
        self,
        sino: int | float | None = None,
        msep=(1, 1),
        edge=0,
        ppm_range: tuple[
            PpmRange | tuple[float | str | ChemicalShift, float | str | ChemicalShift],
            PpmRange | tuple[float | str | ChemicalShift, float | str | ChemicalShift],
        ] = (None, None),
        time_param: str = "delay_time",
        fit_method: Literal["gaussian", "lorentzian"] = "gaussian",
    ) -> PeakList:
        return spectrum_peak_picking.pick_pseudo_nd(
            self,
            sino=sino,
            msep=msep,
            edge=edge,
            ppm_range=ppm_range,
            time_param=time_param,
            fit_method=fit_method,
        )
