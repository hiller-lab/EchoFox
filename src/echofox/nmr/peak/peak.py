import numpy as np

from echofox.nmr.chemical_shift import ChemicalShift

from .exceptions import (
    DimensionMismatchError,
    InvalidPeakDimensionError,
    InvalidPeakIntensityError,
    NucleusMismatchError,
    PeakError,
)


class NmrPeak:
    """
    Represents an n-dimensional NMR peak.

    The Peak class supports peaks from any NMR experiment dimension (1D, 2D, 3D, etc.).
    Each dimension has a chemical shift value, and can have associated nucleus type and
    frequency information. The peak can have additional properties like intensity, volume,
    and linewidths.

    Args:
        chemical_shifts: List of chemical shifts (as floats, ints, strings, or ChemicalShift objects)
        nuclei: List of nucleus labels for each dimension (e.g., '1H', '13C', '15N') (optional)
        frequencies: List of spectrometer frequencies in MHz for each dimension (optional)
        intensity: Peak intensity/height (optional)
        volume: Peak volume/integral (optional)
        linewidths: List of linewidths for each dimension in Hz (optional)
        gaussian_sigmas: Gaussian contributions (sigma) per dimension for Voigt profiles (optional)
        lorentzian_gammas: Lorentzian contributions (gamma) per dimension for Voigt profiles (optional)
        assignments: List of assignment labels for each dimension (e.g., ['H13', 'N13', 'Ca13']) (optional)
        **kwargs: Additional peak properties
    """

    def __init__(
        self,
        chemical_shifts: list[float | int | str | ChemicalShift],
        nuclei: list[str] | None = None,
        frequencies: list[float] | None = None,
        intensity: float | None = None,
        volume: float | None = None,
        linewidths: list[float] | None = None,
        assignments: list[str] | None = None,
        gaussian_sigmas: list[float] | None = None,
        lorentzian_gammas: list[float] | None = None,
        **kwargs,
    ):
        if not chemical_shifts:
            raise InvalidPeakDimensionError("Peak must have at least one dimension")

        # Convert all chemical shifts to ChemicalShift objects
        self._chemical_shifts = []
        for cs in chemical_shifts:
            if isinstance(cs, ChemicalShift):
                self._chemical_shifts.append(cs)
            else:
                self._chemical_shifts.append(ChemicalShift(cs))

        self._ndim = len(self._chemical_shifts)

        # Validate and set nuclei
        if nuclei is not None:
            if len(nuclei) != self._ndim:
                raise InvalidPeakDimensionError(
                    f"Number of nuclei ({len(nuclei)}) must match number of dimensions ({self._ndim})"
                )
            self._nuclei = [str(n) for n in nuclei]
        else:
            self._nuclei = None

        # Validate and set frequencies
        if frequencies is not None:
            if len(frequencies) != self._ndim:
                raise InvalidPeakDimensionError(
                    f"Number of frequencies ({len(frequencies)}) must match number of dimensions ({self._ndim})"
                )
            self._frequencies = [float(f) for f in frequencies]
        else:
            self._frequencies = None

        # Validate and set intensity
        if intensity is not None:
            if not isinstance(intensity, (int, float)):
                raise InvalidPeakIntensityError(intensity)
            self._intensity = float(intensity)
        else:
            self._intensity = None

        # Validate and set volume
        if volume is not None:
            if not isinstance(volume, (int, float)):
                raise InvalidPeakIntensityError(volume)
            self._volume = float(volume)
        else:
            self._volume = None

        # Validate and set linewidths
        if linewidths is not None:
            if len(linewidths) != self._ndim:
                raise InvalidPeakDimensionError(
                    f"Number of linewidths ({len(linewidths)}) must match number of dimensions ({self._ndim})"
                )
            self._linewidths = [float(lw) for lw in linewidths]
        else:
            self._linewidths = None

        # Validate and set assignments
        if assignments is not None:
            if len(assignments) != self._ndim:
                raise InvalidPeakDimensionError(
                    f"Number of assignments ({len(assignments)}) must match number of dimensions ({self._ndim})"
                )
            self._assignments = [str(a) if a is not None else None for a in assignments]
        else:
            self._assignments = None

        # Validate and set Gaussian sigma widths
        if gaussian_sigmas is not None:
            if len(gaussian_sigmas) != self._ndim:
                raise InvalidPeakDimensionError(
                    f"Number of Gaussian sigmas ({len(gaussian_sigmas)}) must match number of dimensions ({self._ndim})"
                )
            self._gaussian_sigmas = [float(s) for s in gaussian_sigmas]
        else:
            self._gaussian_sigmas = None

        # Validate and set Lorentzian gamma widths
        if lorentzian_gammas is not None:
            if len(lorentzian_gammas) != self._ndim:
                raise InvalidPeakDimensionError(
                    f"Number of Lorentzian gammas ({len(lorentzian_gammas)}) must match number of dimensions "
                    f"({self._ndim})"
                )
            self._lorentzian_gammas = [float(g) for g in lorentzian_gammas]
        else:
            self._lorentzian_gammas = None

        # Store extra properties
        self._extra_properties = {}
        for key, value in kwargs.items():
            # create a private attribute to store the value
            setattr(self, "_" + key, value)
            self._extra_properties[key] = value

            # add a property to the class if it doesn't exist
            if not hasattr(self.__class__, key):
                setattr(self.__class__, key, self.make_property(key))

    def make_property(self, attr):
        def getter(self):
            return getattr(self, "_" + attr)

        def setter(self, value):
            setattr(self, "_" + attr, value)

        return property(getter, setter)

    @property
    def chemical_shifts(self) -> list[ChemicalShift]:
        """Returns list of ChemicalShift objects for each dimension."""
        return self._chemical_shifts.copy()

    @property
    def ppm_values(self) -> list[float]:
        """Returns list of chemical shift values in ppm."""
        return [cs.ppm for cs in self._chemical_shifts]

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions."""
        return self._ndim

    @property
    def nuclei(self) -> list[str] | None:
        """Returns the nucleus labels for each dimension."""
        return self._nuclei.copy() if self._nuclei is not None else None

    @nuclei.setter
    def nuclei(self, value: list[str] | None) -> None:
        """Sets the nucleus labels for each dimension."""
        if value is not None:
            if len(value) != self._ndim:
                raise InvalidPeakDimensionError(
                    f"Number of nuclei ({len(value)}) must match number of dimensions ({self._ndim})"
                )
            self._nuclei = [str(n) for n in value]
        else:
            self._nuclei = None

    @property
    def frequencies(self) -> list[float] | None:
        """Returns the spectrometer frequencies in MHz for each dimension."""
        return self._frequencies.copy() if self._frequencies is not None else None

    @frequencies.setter
    def frequencies(self, value: list[float] | None) -> None:
        """Sets the spectrometer frequencies in MHz for each dimension."""
        if value is not None:
            if len(value) != self._ndim:
                raise InvalidPeakDimensionError(
                    f"Number of frequencies ({len(value)}) must match number of dimensions ({self._ndim})"
                )
            self._frequencies = [float(f) for f in value]
        else:
            self._frequencies = None

    @property
    def intensity(self) -> float | None:
        """Returns the peak intensity."""
        return self._intensity

    @intensity.setter
    def intensity(self, value: float | None) -> None:
        """Sets the peak intensity."""
        if value is not None and not isinstance(value, (int, float)):
            raise InvalidPeakIntensityError(value)
        self._intensity = float(value) if value is not None else None

    @property
    def volume(self) -> float | None:
        """Returns the peak volume."""
        return self._volume

    @volume.setter
    def volume(self, value: float | None) -> None:
        """Sets the peak volume."""
        if value is not None and not isinstance(value, (int, float)):
            raise InvalidPeakIntensityError(value)
        self._volume = float(value) if value is not None else None

    @property
    def linewidths(self) -> list[float] | None:
        """Returns the linewidths for each dimension."""
        return self._linewidths.copy() if self._linewidths is not None else None

    @linewidths.setter
    def linewidths(self, value: list[float] | None) -> None:
        """Sets the linewidths for each dimension."""
        if value is not None:
            if len(value) != self._ndim:
                raise InvalidPeakDimensionError(
                    f"Number of linewidths ({len(value)}) must match number of dimensions ({self._ndim})"
                )
            self._linewidths = [float(lw) for lw in value]
        else:
            self._linewidths = None

    @property
    def assignments(self) -> list[str] | None:
        """Returns the assignment labels for each dimension."""
        return self._assignments.copy() if self._assignments is not None else None

    @assignments.setter
    def assignments(self, value: list[str] | None) -> None:
        """Sets the assignment labels for each dimension."""
        if value is not None:
            if len(value) != self._ndim:
                raise InvalidPeakDimensionError(
                    f"Number of assignments ({len(value)}) must match number of dimensions ({self._ndim})"
                )
            self._assignments = [str(a) if a is not None else None for a in value]
        else:
            self._assignments = None

    @property
    def gaussian_sigmas(self) -> list[float] | None:
        """Returns Gaussian width (sigma) values for each dimension."""
        return self._gaussian_sigmas.copy() if self._gaussian_sigmas is not None else None

    @gaussian_sigmas.setter
    def gaussian_sigmas(self, value: list[float] | None) -> None:
        """Sets Gaussian width (sigma) values for each dimension."""
        if value is not None:
            if len(value) != self._ndim:
                raise InvalidPeakDimensionError(
                    f"Number of Gaussian sigmas ({len(value)}) must match number of dimensions ({self._ndim})"
                )
            self._gaussian_sigmas = [float(s) for s in value]
        else:
            self._gaussian_sigmas = None

    @property
    def lorentzian_gammas(self) -> list[float] | None:
        """Returns Lorentzian width (gamma) values for each dimension."""
        return self._lorentzian_gammas.copy() if self._lorentzian_gammas is not None else None

    @lorentzian_gammas.setter
    def lorentzian_gammas(self, value: list[float] | None) -> None:
        """Sets Lorentzian width (gamma) values for each dimension."""
        if value is not None:
            if len(value) != self._ndim:
                raise InvalidPeakDimensionError(
                    f"Number of Lorentzian gammas ({len(value)}) must match number of dimensions ({self._ndim})"
                )
            self._lorentzian_gammas = [float(g) for g in value]
        else:
            self._lorentzian_gammas = None

    def get_chemical_shift(self, dimension: int) -> ChemicalShift:
        """
        Returns the ChemicalShift object for a specific dimension.

        Args:
            dimension: Dimension index (0-based)

        Returns:
            ChemicalShift object for the specified dimension

        Raises:
            IndexError: If dimension is out of range
        """
        if dimension < 0 or dimension >= self._ndim:
            raise IndexError(f"Dimension {dimension} is out of range for {self._ndim}D peak")
        return self._chemical_shifts[dimension]

    def set_chemical_shift(self, dimension: int, value: float | int | str | ChemicalShift) -> None:
        """
        Sets the chemical shift for a specific dimension.

        Args:
            dimension: Dimension index (0-based)
            value: New chemical shift value

        Raises:
            IndexError: If dimension is out of range
        """
        if dimension < 0 or dimension >= self._ndim:
            raise IndexError(f"Dimension {dimension} is out of range for {self._ndim}D peak")

        if isinstance(value, ChemicalShift):
            self._chemical_shifts[dimension] = value
        else:
            self._chemical_shifts[dimension] = ChemicalShift(value)

    def get_nucleus(self, dimension: int) -> str | None:
        """
        Returns the nucleus label for a specific dimension.

        Args:
            dimension: Dimension index (0-based)

        Returns:
            Nucleus label (e.g., '1H', '13C') or None if not set

        Raises:
            IndexError: If dimension is out of range
        """
        if dimension < 0 or dimension >= self._ndim:
            raise IndexError(f"Dimension {dimension} is out of range for {self._ndim}D peak")
        return self._nuclei[dimension] if self._nuclei is not None else None

    def set_nucleus(self, dimension: int, value: str) -> None:
        """
        Sets the nucleus label for a specific dimension.

        Args:
            dimension: Dimension index (0-based)
            value: Nucleus label (e.g., '1H', '13C')

        Raises:
            IndexError: If dimension is out of range
        """
        if dimension < 0 or dimension >= self._ndim:
            raise IndexError(f"Dimension {dimension} is out of range for {self._ndim}D peak")

        if self._nuclei is None:
            self._nuclei = [None] * self._ndim

        self._nuclei[dimension] = str(value)

    def get_frequency(self, dimension: int) -> float | None:
        """
        Returns the spectrometer frequency for a specific dimension.

        Args:
            dimension: Dimension index (0-based)

        Returns:
            Frequency in MHz or None if not set

        Raises:
            IndexError: If dimension is out of range
        """
        if dimension < 0 or dimension >= self._ndim:
            raise IndexError(f"Dimension {dimension} is out of range for {self._ndim}D peak")
        return self._frequencies[dimension] if self._frequencies is not None else None

    def set_frequency(self, dimension: int, value: float) -> None:
        """
        Sets the spectrometer frequency for a specific dimension.

        Args:
            dimension: Dimension index (0-based)
            value: Frequency in MHz

        Raises:
            IndexError: If dimension is out of range
        """
        if dimension < 0 or dimension >= self._ndim:
            raise IndexError(f"Dimension {dimension} is out of range for {self._ndim}D peak")

        if self._frequencies is None:
            self._frequencies = [None] * self._ndim

        self._frequencies[dimension] = float(value)

    def get_assignment(self, dimension: int) -> str | None:
        """
        Returns the assignment label for a specific dimension.

        Args:
            dimension: Dimension index (0-based)

        Returns:
            Assignment label (e.g., 'H13', 'N13') or None if not set

        Raises:
            IndexError: If dimension is out of range
        """
        if dimension < 0 or dimension >= self._ndim:
            raise IndexError(f"Dimension {dimension} is out of range for {self._ndim}D peak")
        return self._assignments[dimension] if self._assignments is not None else None

    def set_assignment(self, dimension: int, value: str | None) -> None:
        """
        Sets the assignment label for a specific dimension.

        Args:
            dimension: Dimension index (0-based)
            value: Assignment label (e.g., 'H13', 'Ca13')

        Raises:
            IndexError: If dimension is out of range
        """
        if dimension < 0 or dimension >= self._ndim:
            raise IndexError(f"Dimension {dimension} is out of range for {self._ndim}D peak")

        if self._assignments is None:
            self._assignments = [None] * self._ndim

        self._assignments[dimension] = str(value) if value is not None else None

    def distance_to(self, other: "NmrPeak", weights: list[float] | None = None) -> float:
        """
        Calculates the Euclidean distance to another peak in chemical shift space.

        Args:
            other: Another Peak object
            weights: Optional weights for each dimension (e.g., to account for different ppm scales)

        Returns:
            Euclidean distance in ppm

        Raises:
            DimensionMismatchError: If peaks have different dimensions
            NucleusMismatchError: If peaks have different nuclei in any dimension
        """
        if self._ndim != other._ndim:
            raise DimensionMismatchError(self._ndim, other._ndim)

        # Check if nuclei match when both peaks have nuclei defined
        if self._nuclei is not None and other._nuclei is not None:
            for i, (nuc1, nuc2) in enumerate(zip(self._nuclei, other._nuclei, strict=True)):
                if nuc1 != nuc2:
                    raise NucleusMismatchError(self._nuclei, other._nuclei, dimension=i)

        if weights is None:
            weights = [1.0] * self._ndim
        elif len(weights) != self._ndim:
            raise InvalidPeakDimensionError(
                f"Number of weights ({len(weights)}) must match number of dimensions ({self._ndim})"
            )

        squared_diff = sum(
            (w * (cs1.ppm - cs2.ppm)) ** 2
            for w, cs1, cs2 in zip(weights, self._chemical_shifts, other._chemical_shifts, strict=True)
        )
        return np.sqrt(squared_diff)

    def is_within_tolerance(self, other: "NmrPeak", tolerances: float | list[float]) -> bool:
        """
        Checks if another peak is within specified tolerance(s) in each dimension.

        Args:
            other: Another Peak object
            tolerances: Single tolerance (applied to all dimensions) or list of tolerances per dimension

        Returns:
            True if peaks are within tolerance in all dimensions

        Raises:
            DimensionMismatchError: If peaks have different dimensions
            NucleusMismatchError: If peaks have different nuclei in any dimension
        """
        if self._ndim != other._ndim:
            raise DimensionMismatchError(self._ndim, other._ndim)

        # Check if nuclei match when both peaks have nuclei defined
        if self._nuclei is not None and other._nuclei is not None:
            for i, (nuc1, nuc2) in enumerate(zip(self._nuclei, other._nuclei, strict=True)):
                if nuc1 != nuc2:
                    raise NucleusMismatchError(self._nuclei, other._nuclei, dimension=i)

        if isinstance(tolerances, (int, float)):
            tolerances = [float(tolerances)] * self._ndim
        elif len(tolerances) != self._ndim:
            raise InvalidPeakDimensionError(
                f"Number of tolerances ({len(tolerances)}) must match number of dimensions ({self._ndim})"
            )

        for cs1, cs2, tol in zip(self._chemical_shifts, other._chemical_shifts, tolerances, strict=True):
            if abs(cs1.ppm - cs2.ppm) > tol:
                return False
        return True

    def __str__(self) -> str:
        """Returns string representation of the peak."""
        # Build dimension strings with nucleus and assignment info if available
        dim_strs = []
        for i, cs in enumerate(self._chemical_shifts):
            dim_str = f"{cs.ppm:.3f}"

            # Add nucleus in parentheses if available
            if self._nuclei is not None:
                dim_str += f"({self._nuclei[i]})"

            # Add assignment if available
            if self._assignments is not None and self._assignments[i] is not None:
                dim_str += f"[{self._assignments[i]}]"

            dim_strs.append(dim_str)

        shifts_str = " | ".join(dim_strs)
        result = f"Peak({shifts_str} ppm)"

        if self._intensity is not None:
            result += f" I={self._intensity:.2e}"

        return result

    def __repr__(self) -> str:
        """Returns detailed representation of the peak."""
        shifts = [cs.ppm for cs in self._chemical_shifts]
        parts = [f"chemical_shifts={shifts}"]

        if self._nuclei is not None:
            parts.append(f"nuclei={self._nuclei}")
        if self._frequencies is not None:
            parts.append(f"frequencies={self._frequencies}")
        if self._intensity is not None:
            parts.append(f"intensity={self._intensity}")
        if self._volume is not None:
            parts.append(f"volume={self._volume}")
        if self._linewidths is not None:
            parts.append(f"linewidths={self._linewidths}")
        if self._assignments is not None:
            parts.append(f"assignments={self._assignments}")
        if self._gaussian_sigmas is not None:
            parts.append(f"gaussian_sigmas={self._gaussian_sigmas}")
        if self._lorentzian_gammas is not None:
            parts.append(f"lorentzian_gammas={self._lorentzian_gammas}")

        return f"Peak({', '.join(parts)})"

    def __eq__(self, other) -> bool:
        """
        Checks equality with another Peak.

        Two peaks are equal if they have the same dimensions and chemical shifts.
        """
        if not isinstance(other, NmrPeak):
            return False

        if self._ndim != other._ndim:
            return False

        return all(cs1 == cs2 for cs1, cs2 in zip(self._chemical_shifts, other._chemical_shifts, strict=True))

    def __hash__(self) -> int:
        """Returns hash of the peak based on chemical shifts."""
        return hash(tuple(cs.ppm for cs in self._chemical_shifts))

    def __getitem__(self, dimension: int) -> ChemicalShift:
        """
        Allows indexing to get chemical shift for a dimension.

        Example:
            peak[0]  # Get chemical shift for first dimension
        """
        return self.get_chemical_shift(dimension)

    def __setitem__(self, dimension: int, value: float | int | str | ChemicalShift) -> None:
        """
        Allows indexing to set chemical shift for a dimension.

        Example:
            peak[0] = 7.26  # Set chemical shift for first dimension
        """
        self.set_chemical_shift(dimension, value)

    def __len__(self) -> int:
        """Returns the number of dimensions."""
        return self._ndim

    def to_dict(self) -> dict:
        """
        Converts the peak to a dictionary representation.

        Returns:
            Dictionary containing all peak properties
        """
        result = {"chemical_shifts": self.ppm_values, "ndim": self._ndim}

        if self._nuclei is not None:
            result["nuclei"] = self._nuclei
        if self._frequencies is not None:
            result["frequencies"] = self._frequencies
        if self._intensity is not None:
            result["intensity"] = self._intensity
        if self._volume is not None:
            result["volume"] = self._volume
        if self._linewidths is not None:
            result["linewidths"] = self._linewidths
        if self._assignments is not None:
            result["assignments"] = self._assignments
        if self._gaussian_sigmas is not None:
            result["gaussian_sigmas"] = self._gaussian_sigmas
        if self._lorentzian_gammas is not None:
            result["lorentzian_gammas"] = self._lorentzian_gammas

        # Serialize extra properties
        for key, value in self._extra_properties.items():
            # Handle PeakAdjustmentFitResult objects
            if hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            else:
                result[key] = value

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "NmrPeak":
        """
        Creates a Peak from a dictionary representation.

        Args:
            data: Dictionary containing peak properties

        Returns:
            Peak object
        """
        try:
            chemical_shifts = data["chemical_shifts"]
            nuclei = data.get("nuclei")
            frequencies = data.get("frequencies")
            intensity = data.get("intensity")
            volume = data.get("volume")
            linewidths = data.get("linewidths")
            assignments = data.get("assignments")
            gaussian_sigmas = data.get("gaussian_sigmas")
            lorentzian_gammas = data.get("lorentzian_gammas")
        except KeyError as e:
            raise PeakError(f"Missing required property '{e.args[0]}'") from e

        # Extract extra properties
        extra_keys = {
            "chemical_shifts",
            "ndim",
            "nuclei",
            "frequencies",
            "intensity",
            "volume",
            "linewidths",
            "assignments",
            "gaussian_sigmas",
            "lorentzian_gammas",
        }
        kwargs = {}
        for k, v in data.items():
            if k not in extra_keys:
                kwargs[k] = v

        return cls(
            chemical_shifts=chemical_shifts,
            nuclei=nuclei,
            frequencies=frequencies,
            intensity=intensity,
            volume=volume,
            linewidths=linewidths,
            assignments=assignments,
            gaussian_sigmas=gaussian_sigmas,
            lorentzian_gammas=lorentzian_gammas,
            **kwargs,
        )
