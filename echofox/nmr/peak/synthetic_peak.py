"""Synthetic peak class for generating n-dimensional Voigt profiles"""

import numpy as np
from scipy.special import wofz

from echofox.nmr.chemical_shift import ChemicalShift

from .exceptions import InvalidPeakDimensionError, InvalidPeakIntensityError
from .peak import NmrPeak


class SyntheticPeak(NmrPeak):
    """
    Represents a synthetic n-dimensional NMR peak with Voigt profile parameters.

    The SyntheticPeak class extends the Peak class to include parameters needed
    for calculating Voigt profiles. A Voigt profile is a convolution of Gaussian
    and Lorentzian lineshapes, commonly used in NMR spectroscopy.

    For each dimension, the peak has:
    - Position (chemical shift)
    - Gaussian width (sigma)
    - Lorentzian width (gamma)

    The peak also has an overall amplitude/intensity.

    Args:
        chemical_shifts: List of chemical shifts (peak centers) in ppm
        amplitude: Peak amplitude/intensity
        gaussian_widths: List of Gaussian width parameters (sigma) for each dimension in ppm
        lorentzian_widths: List of Lorentzian width parameters (gamma) for each dimension in ppm
        nuclei: List of nucleus labels for each dimension (e.g., '1H', '13C', '15N') (optional)
        frequencies: List of spectrometer frequencies in MHz for each dimension (optional)
        assignments: List of assignment labels for each dimension (optional)
        **kwargs: Additional peak properties

    Example:
        # Create a 2D synthetic peak for 1H-15N HSQC
        peak = SyntheticPeak(
            chemical_shifts=[8.5, 120.0],
            amplitude=1000.0,
            gaussian_widths=[0.005, 0.15],
            lorentzian_widths=[0.003, 0.10],
            nuclei=['1H', '15N'],
            frequencies=[600.0, 60.8]
        )

        # Evaluate the Voigt profile at specific coordinates
        h1_coords = np.linspace(8.4, 8.6, 100)
        n15_coords = np.linspace(119.5, 120.5, 100)
        H1, N15 = np.meshgrid(h1_coords, n15_coords)
        intensity = peak.evaluate([H1, N15])
    """

    def __init__(
        self,
        chemical_shifts: list[float | int | str | ChemicalShift],
        amplitude: float,
        gaussian_widths: list[float],
        lorentzian_widths: list[float],
        nuclei: list[str] | None = None,
        frequencies: list[float] | None = None,
        assignments: list[str] | None = None,
        **kwargs,
    ):
        # Initialize the base Peak class
        super().__init__(
            chemical_shifts=chemical_shifts,
            nuclei=nuclei,
            frequencies=frequencies,
            intensity=amplitude,
            assignments=assignments,
            **kwargs,
        )

        # Validate amplitude
        if not isinstance(amplitude, (int, float)):
            raise InvalidPeakIntensityError(amplitude)
        self._amplitude = float(amplitude)

        # Validate and set Gaussian widths
        if len(gaussian_widths) != self._ndim:
            raise InvalidPeakDimensionError(
                f"Number of Gaussian widths ({len(gaussian_widths)}) must match "
                f"number of dimensions ({self._ndim})"
            )
        self._gaussian_widths = [float(sigma) for sigma in gaussian_widths]

        # Validate and set Lorentzian widths
        if len(lorentzian_widths) != self._ndim:
            raise InvalidPeakDimensionError(
                f"Number of Lorentzian widths ({len(lorentzian_widths)}) must match "
                f"number of dimensions ({self._ndim})"
            )
        self._lorentzian_widths = [float(gamma) for gamma in lorentzian_widths]

    @property
    def amplitude(self) -> float:
        """Returns the peak amplitude."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        """Sets the peak amplitude."""
        if not isinstance(value, (int, float)):
            raise InvalidPeakIntensityError(value)
        self._amplitude = float(value)
        # Also update the intensity in the base class
        self._intensity = float(value)

    @property
    def gaussian_widths(self) -> list[float]:
        """Returns the Gaussian width parameters (sigma) for each dimension."""
        return self._gaussian_widths.copy()

    @gaussian_widths.setter
    def gaussian_widths(self, value: list[float]) -> None:
        """Sets the Gaussian width parameters for each dimension."""
        if len(value) != self._ndim:
            raise InvalidPeakDimensionError(
                f"Number of Gaussian widths ({len(value)}) must match "
                f"number of dimensions ({self._ndim})"
            )
        self._gaussian_widths = [float(sigma) for sigma in value]

    @property
    def lorentzian_widths(self) -> list[float]:
        """Returns the Lorentzian width parameters (gamma) for each dimension."""
        return self._lorentzian_widths.copy()

    @lorentzian_widths.setter
    def lorentzian_widths(self, value: list[float]) -> None:
        """Sets the Lorentzian width parameters for each dimension."""
        if len(value) != self._ndim:
            raise InvalidPeakDimensionError(
                f"Number of Lorentzian widths ({len(value)}) must match "
                f"number of dimensions ({self._ndim})"
            )
        self._lorentzian_widths = [float(gamma) for gamma in value]

    @staticmethod
    def _voigt_profile_1d(
        x: np.ndarray, center: float, sigma: float, gamma: float
    ) -> np.ndarray:
        """
        Calculate unnormalized 1D Voigt profile.

        Args:
            x: Coordinates at which to evaluate the profile
            center: Peak center position
            sigma: Gaussian width parameter
            gamma: Lorentzian width parameter

        Returns:
            Voigt profile values at coordinates x
        """
        # Ensure sigma is not too small to avoid numerical issues
        sigma = max(sigma, 1e-9)
        # Calculate the complex argument for the Faddeeva function
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        # Return the real part of the Faddeeva function
        return np.real(wofz(z))

    def evaluate(self, coordinates: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """
        Evaluate the n-dimensional Voigt profile at given coordinates.

        The n-dimensional Voigt profile is calculated as the product of 1D Voigt
        profiles in each dimension, scaled by the amplitude.

        Args:
            coordinates: List of coordinate arrays, one for each dimension.
                        For 1D: single array of shape (N,)
                        For 2D: list of two arrays of shape (M, N) from meshgrid
                        For 3D: list of three arrays of shape (L, M, N) from meshgrid
                        etc.

        Returns:
            Voigt profile values at the given coordinates

        Example:
            # 1D example
            x = np.linspace(8.0, 9.0, 100)
            intensity = peak.evaluate(x)

            # 2D example
            h1 = np.linspace(8.0, 9.0, 100)
            n15 = np.linspace(115.0, 125.0, 100)
            H1, N15 = np.meshgrid(h1, n15)
            intensity = peak.evaluate([H1, N15])
        """
        # Handle single array input for 1D case
        if isinstance(coordinates, np.ndarray):
            if self._ndim != 1:
                raise ValueError(
                    f"For {self._ndim}D peak, provide a list of {self._ndim} coordinate arrays"
                )
            coordinates = [coordinates]

        # Validate number of coordinate arrays
        if len(coordinates) != self._ndim:
            raise ValueError(
                f"Number of coordinate arrays ({len(coordinates)}) must match "
                f"peak dimensions ({self._ndim})"
            )

        # Start with amplitude
        result = self._amplitude

        # Multiply by 1D Voigt profile for each dimension
        for i, coord in enumerate(coordinates):
            center = self._chemical_shifts[i].ppm
            sigma = self._gaussian_widths[i]
            gamma = self._lorentzian_widths[i]
            result = result * self._voigt_profile_1d(coord, center, sigma, gamma)

        return result

    def evaluate_at_points(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate the Voigt profile at specific points in n-dimensional space.

        Args:
            points: Array of shape (N, ndim) where N is the number of points
                   and ndim is the number of dimensions. Each row is a point
                   at which to evaluate the profile.

        Returns:
            Array of shape (N,) with the Voigt profile values

        Example:
            # Evaluate at specific points
            points = np.array([
                [8.5, 120.0],
                [8.51, 120.1],
                [8.49, 119.9]
            ])
            intensities = peak.evaluate_at_points(points)
        """
        if points.ndim != 2:
            raise ValueError("Points must be a 2D array of shape (N, ndim)")

        if points.shape[1] != self._ndim:
            raise ValueError(
                f"Points must have {self._ndim} columns to match peak dimensions, "
                f"got {points.shape[1]}"
            )

        # Start with amplitude
        result = np.full(points.shape[0], self._amplitude)

        # Multiply by 1D Voigt profile for each dimension
        for i in range(self._ndim):
            center = self._chemical_shifts[i].ppm
            sigma = self._gaussian_widths[i]
            gamma = self._lorentzian_widths[i]
            result = result * self._voigt_profile_1d(points[:, i], center, sigma, gamma)

        return result

    def get_gaussian_width(self, dimension: int) -> float:
        """
        Get the Gaussian width parameter for a specific dimension.

        Args:
            dimension: Dimension index (0-based)

        Returns:
            Gaussian width (sigma) for the specified dimension
        """
        if dimension < 0 or dimension >= self._ndim:
            raise IndexError(
                f"Dimension {dimension} is out of range for {self._ndim}D peak"
            )
        return self._gaussian_widths[dimension]

    def set_gaussian_width(self, dimension: int, value: float) -> None:
        """
        Set the Gaussian width parameter for a specific dimension.

        Args:
            dimension: Dimension index (0-based)
            value: New Gaussian width (sigma) value
        """
        if dimension < 0 or dimension >= self._ndim:
            raise IndexError(
                f"Dimension {dimension} is out of range for {self._ndim}D peak"
            )
        self._gaussian_widths[dimension] = float(value)

    def get_lorentzian_width(self, dimension: int) -> float:
        """
        Get the Lorentzian width parameter for a specific dimension.

        Args:
            dimension: Dimension index (0-based)

        Returns:
            Lorentzian width (gamma) for the specified dimension
        """
        if dimension < 0 or dimension >= self._ndim:
            raise IndexError(
                f"Dimension {dimension} is out of range for {self._ndim}D peak"
            )
        return self._lorentzian_widths[dimension]

    def set_lorentzian_width(self, dimension: int, value: float) -> None:
        """
        Set the Lorentzian width parameter for a specific dimension.

        Args:
            dimension: Dimension index (0-based)
            value: New Lorentzian width (gamma) value
        """
        if dimension < 0 or dimension >= self._ndim:
            raise IndexError(
                f"Dimension {dimension} is out of range for {self._ndim}D peak"
            )
        self._lorentzian_widths[dimension] = float(value)

    def to_dict(self) -> dict:
        """
        Convert the synthetic peak to a dictionary representation.

        Returns:
            Dictionary containing all peak properties including Voigt parameters
        """
        result = super().to_dict()
        result["amplitude"] = self._amplitude
        result["gaussian_widths"] = self._gaussian_widths
        result["lorentzian_widths"] = self._lorentzian_widths
        result["peak_type"] = "synthetic_voigt"
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SyntheticPeak":
        """
        Create a SyntheticPeak from a dictionary representation.

        Args:
            data: Dictionary containing peak properties

        Returns:
            SyntheticPeak object
        """
        chemical_shifts = data["chemical_shifts"]
        amplitude = data["amplitude"]
        gaussian_widths = data["gaussian_widths"]
        lorentzian_widths = data["lorentzian_widths"]
        nuclei = data.get("nuclei")
        frequencies = data.get("frequencies")
        assignments = data.get("assignments")

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
            "amplitude",
            "gaussian_widths",
            "lorentzian_widths",
            "peak_type",
        }
        kwargs = {}
        for k, v in data.items():
            if k not in extra_keys:
                kwargs[k] = v

        return cls(
            chemical_shifts=chemical_shifts,
            amplitude=amplitude,
            gaussian_widths=gaussian_widths,
            lorentzian_widths=lorentzian_widths,
            nuclei=nuclei,
            frequencies=frequencies,
            assignments=assignments,
            **kwargs,
        )

    @classmethod
    def from_peak(
        cls,
        peak: NmrPeak,
        amplitude: float | None = None,
        gaussian_widths: list[float] | None = None,
        lorentzian_widths: list[float] | None = None,
    ) -> "SyntheticPeak":
        """
        Create a SyntheticPeak from an existing Peak object.

        Args:
            peak: Existing Peak object
            amplitude: Peak amplitude (defaults to peak.intensity or 1.0)
            gaussian_widths: Gaussian widths (defaults to 0.01 ppm for each dimension)
            lorentzian_widths: Lorentzian widths (defaults to 0.005 ppm for each dimension)

        Returns:
            SyntheticPeak object with same positions as input peak
        """
        # Get amplitude from peak intensity or default
        if amplitude is None:
            amplitude = peak.intensity if peak.intensity is not None else 1.0

        # Generate default widths if not provided
        if gaussian_widths is None:
            # Use linewidths from peak if available, otherwise use defaults
            if peak.linewidths is not None:
                # Convert FWHM to sigma (for Gaussian: FWHM ≈ 2.355 * sigma)
                gaussian_widths = [lw / 2.355 for lw in peak.linewidths]
            else:
                gaussian_widths = [0.01] * peak.ndim

        if lorentzian_widths is None:
            # Default to half of Gaussian width
            lorentzian_widths = [sigma / 2 for sigma in gaussian_widths]

        return cls(
            chemical_shifts=peak.ppm_values,
            amplitude=amplitude,
            gaussian_widths=gaussian_widths,
            lorentzian_widths=lorentzian_widths,
            nuclei=peak.nuclei,
            frequencies=peak.frequencies,
            assignments=peak.assignments,
        )

    def __str__(self) -> str:
        """Return string representation of the synthetic peak."""
        base_str = super().__str__()
        voigt_info = f" Voigt(σ={self._gaussian_widths}, γ={self._lorentzian_widths})"
        return base_str + voigt_info

    def __repr__(self) -> str:
        """Return detailed representation of the synthetic peak."""
        shifts = [cs.ppm for cs in self._chemical_shifts]
        parts = [
            f"chemical_shifts={shifts}",
            f"amplitude={self._amplitude}",
            f"gaussian_widths={self._gaussian_widths}",
            f"lorentzian_widths={self._lorentzian_widths}",
        ]

        if self._nuclei is not None:
            parts.append(f"nuclei={self._nuclei}")
        if self._frequencies is not None:
            parts.append(f"frequencies={self._frequencies}")
        if self._assignments is not None:
            parts.append(f"assignments={self._assignments}")

        return f"SyntheticPeak({', '.join(parts)})"
