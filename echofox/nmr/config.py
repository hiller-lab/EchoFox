"""
Package-wide configuration and default values for MiraNMR.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class NmrConfig:
    """
    Configuration settings for MiraNMR package.

    Modify the global `config` instance to change default behavior:

    Examples:
        from echofox.nmr.config import config

        # Change default spectrum format
        config.default_spectrum_format = 'pipe'

        # Disable chemical shift validation
        config.validate_chemical_shifts = False

        # Reset to defaults
        from echofox.nmr.config import reset_config
        reset_config()
    """

    # -------------------------------------------------------------------------
    # Chemical shift settings
    # -------------------------------------------------------------------------

    # Validation range for chemical shifts (ppm)
    chemical_shift_min: float = -50.0
    chemical_shift_max: float = 300.0

    # Whether to validate chemical shift values are within typical NMR range
    validate_chemical_shifts: bool = True

    # -------------------------------------------------------------------------
    # Spectrum settings
    # -------------------------------------------------------------------------

    # Default file format for loading spectra
    default_spectrum_format: Literal["bruker", "pipe"] = "bruker"

    # Default noise estimation method ('std' or 'median_absolute')
    default_noise_method: Literal["std", "median_absolute"] = "median_absolute"

    # Default minimum signal-to-noise ratio for peak picking
    default_min_sino: float = 8.0

    # Default tolerance for ppm matching (in ppm)
    default_ppm_tolerance: float = 0.1
    default_ppm_tolerance_15n: float = 0.3
    default_ppm_tolerance_13c: float = 0.3
    default_ppm_tolerance_1h: float = 0.1

    # Default tolerance for time matching (in seconds)
    default_time_tolerance: float = 1e-3

    # -------------------------------------------------------------------------
    # Peak picking settings
    # -------------------------------------------------------------------------

    # Default tolerance for peak matching (in ppm)
    default_peak_tolerance: float = 0.01

    # Minimum peak intensity relative to noise
    default_peak_threshold: float = 5.0

    # Peak adjustment algorithm
    peak_picking_adjust: str = "gaussian"

    # -------------------------------------------------------------------------
    # File patterns
    # -------------------------------------------------------------------------

    # Glob patterns for finding spectrum files
    bruker_1d_pattern: str = "1r"
    bruker_2d_pattern: str = "2rr"
    pipe_1d_pattern: str = "*.ft1"
    pipe_2d_pattern: str = "*.ft2"
    pipe_pattern: str = "*.ft*"

    # -------------------------------------------------------------------------
    # Display and formatting settings
    # -------------------------------------------------------------------------

    # Decimal places for ppm values
    ppm_decimal_places: int = 3

    # Decimal places for intensity values
    intensity_decimal_places: int = 2

    # Decimal places for frequency values (MHz)
    frequency_decimal_places: int = 2

    # -------------------------------------------------------------------------
    # Relaxation analysis settings
    # -------------------------------------------------------------------------

    # Default fitting method for relaxation curves
    default_relaxation_fit_method: str = "leastsq"

    # Maximum iterations for fitting
    max_fit_iterations: int = 1000

    # -------------------------------------------------------------------------
    # Dataset settings
    # -------------------------------------------------------------------------

    # Whether to allow duplicate experiment names in datasets
    allow_duplicate_experiments: bool = False


# Global configuration instance
config = NmrConfig()


def reset_config() -> None:
    """Reset configuration to default values."""
    global config
    config = NmrConfig()


def get_config() -> NmrConfig:
    """Get the current configuration instance."""
    return config
