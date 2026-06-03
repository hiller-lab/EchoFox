"""
Package-wide configuration and default values for echofox-plot.
"""

import os
from dataclasses import dataclass, field


@dataclass
class PlotConfig:
    """
    Configuration settings for echofoxplot package.

    Modify the global `config` instance to change default behavior:

    Examples:
        from echofoxplot.config import config

        # Change default figure size
        config.figure_size = [8.0, 6.0]

        # Change tick spacing for 2D plots
        config.tick_spacing_2d = [[10, 2], [2, 0.5]]

        # Reset to defaults
        from echofoxplot.config import reset_config
        reset_config()
    """

    # -------------------------------------------------------------------------
    # Mira Settings
    # -------------------------------------------------------------------------

    file_path = os.path.dirname(os.path.abspath(__file__))
    ECHOFOX_MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ECHOFOX_EXAMPLE_DATA_DIR = os.path.join(ECHOFOX_MAIN_DIR, "examples", "example_data")

    # -------------------------------------------------------------------------
    # Figure settings
    # -------------------------------------------------------------------------

    # Default figure size [width, height] in inches
    figure_size: list[float] = field(default_factory=lambda: [5.83, 4.13])

    # DPI for figure output
    figure_dpi: int = 150

    # Default figure format for saving
    default_save_format: str = "png"

    # -------------------------------------------------------------------------
    # Tick spacing settings
    # -------------------------------------------------------------------------

    # Tick spacing for 2D plots [[major_f1, minor_f1], [major_f2, minor_f2]]
    tick_spacing_2d: list[list[float]] = field(default_factory=lambda: [[5, 1], [1, 0.2]])

    # Tick spacing for 1D plots [major, minor]
    tick_spacing_1d: list[float] = field(default_factory=lambda: [1, 0.2])

    # -------------------------------------------------------------------------
    # Projection settings
    # -------------------------------------------------------------------------

    # Ratio of projection height to main plot (F1 projection on side)
    projection_f1_ratio: float = 0.85

    # Ratio of projection width to main plot (F2 projection on top)
    projection_f2_ratio: float = 0.85

    # Whether to show projections by default
    show_projections: bool = True

    # -------------------------------------------------------------------------
    # Contour settings
    # -------------------------------------------------------------------------

    # Default number of contour levels
    default_contour_levels: int = 10

    # Default contour level factor (multiplicative)
    contour_level_factor: float = 1.4

    # Default positive contour color
    positive_contour_color: str = "black"

    # Default negative contour color
    negative_contour_color: str = "red"

    # Default contour line width
    contour_linewidth: float = 0.3

    # -------------------------------------------------------------------------
    # 1D plot settings
    # -------------------------------------------------------------------------

    # Default line color for 1D spectra
    default_line_color: str = "black"

    # Default line width for 1D spectra
    default_linewidth: float = 1

    # -------------------------------------------------------------------------
    # Axis and label settings
    # -------------------------------------------------------------------------

    # Font size for axis labels
    axis_label_fontsize: int = 10

    # Font size for tick labels
    tick_label_fontsize: int = 8

    # Font size for title
    title_fontsize: int = 12

    # Whether to invert ppm axis (high to low)
    invert_ppm_axis: bool = True

    # -------------------------------------------------------------------------
    # Peak annotation settings
    # -------------------------------------------------------------------------

    # Default marker for peaks
    peak_marker: str = "x"

    # Default marker size for peaks
    peak_marker_size: float = 5.0

    # Default color for peak markers
    peak_marker_color: str = "red"

    # Font size for peak labels
    peak_label_fontsize: int = 6

    # -------------------------------------------------------------------------
    # Color settings
    # -------------------------------------------------------------------------

    # Default colormap for 2D spectra
    default_colormap: str = "viridis"

    # Background color
    background_color: str = "white"

    color1D: list = field(
        default_factory=lambda: [
            "#0C5DA5",
            "#00B945",
            "#FF9500",
            "#FF2C00",
            "#845B97",
            "#474747",
            "#9e9e9e",
        ]
    )

    color2D: list = field(
        default_factory=lambda: [
            "#0C5DA5",
            "#00B945",
            "#FF9500",
            "#FF2C00",
            "#845B97",
            "#474747",
            "#9e9e9e",
        ]
    )


# Global configuration instance
config = PlotConfig()


def reset_config() -> None:
    """Reset configuration to default values."""
    global config
    config = PlotConfig()


def get_config() -> PlotConfig:
    """Get the current configuration instance."""
    return config
