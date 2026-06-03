# Import standard matplotlib modules for plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes

# Configure font types for PDF and PostScript output to ensure compatibility with Illustrator and others
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["figure.constrained_layout.use"] = True

# Import echofox.core components
from echofox.utils.units import convert_to_inches
from echofox.core.typing import Number
from echofox.core.colors import Color

# Import echofoxplot config
from .config import config


# Import echofoxplot axes
from echofox.echofoxplot.axes.spectrum_axes import SpectrumAxes

# Import plotting functions from echofoxplot
from echofox.echofoxplot.plot.spectrum.spectrum2d import plot2d as _plot2d
from echofox.echofoxplot.plot.spectrum.spectrum1d import plot1d as _plot1d


# ----------------------------------------------------------------------------------------------------------------------
# Helper function to determine and return the current figure and axis to plot on
#


def _get_context(
    current_figure: Figure | None = None,
    current_ax: SpectrumAxes | None = None,
    figure_size: tuple[Number, Number] | tuple[str, str] = None,
    spectrum_axes: bool = True,
):
    """
    Returns a figure and axis object to plot on. If none exist, they are created.
    This function supports both standard and SpectrumAxes (a specialized axis class).
    """

    # Use config default if not specified
    if figure_size is None:
        figure_size = config.figure_size

    # Use axis' figure if axis is provided
    if current_ax is not None:
        figure = current_ax.figure
    else:
        # Use current or create new figure
        if current_figure is None:
            figure = plt.gcf()
            figure.set_size_inches(*map(convert_to_inches, figure_size))
        else:
            figure = current_figure

    # Use or create axis
    if current_ax is None:
        if figure.get_axes() == []:
            if spectrum_axes:
                ax = figure.add_subplot(
                    111, projection="SpectrumAxes"
                )  # Custom projection
            else:
                ax = figure.add_subplot(111)
        else:
            ax = figure.gca()  # Get current axis from figure
    else:
        ax = current_ax

    return figure, ax


def _create_new_context(spectrum_axes: bool = True):
    """
    Returns a figure and axis object to plot on.
    This function supports both standard and SpectrumAxes (a specialized axis class).
    """
    figure = Figure()
    if spectrum_axes:
        ax = figure.add_subplot(111, projection="SpectrumAxes")  # Custom projection
    else:
        ax = figure.add_subplot(111)
    return figure, ax


# ----------------------------------------------------------------------------------------------------------------------
# High-level MiraPlot plotting functions
#


def plot2d(
    *args,
    current_figure: Figure | None = None,
    current_ax: SpectrumAxes | None = None,
    dpi: Number = None,
    figure_size: tuple[Number, Number] | tuple[str, str] = None,
    **kwargs,
):
    """
    Wrapper for 2D spectrum plotting using MiraPlot.
    """
    figure, ax = _get_context(current_figure, current_ax, figure_size)
    figure.set_dpi(dpi if dpi is not None else config.figure_dpi)
    return _plot2d(current_ax=ax, current_figure=figure, *args, **kwargs)


def plot1d(
    *args,
    current_figure: Figure | None = None,
    current_ax: SpectrumAxes | None = None,
    dpi: Number = None,
    figure_size: tuple[Number, Number] | tuple[str, str] = None,
    **kwargs,
):
    """
    Wrapper for 1D spectrum plotting using MiraPlot.
    """
    figure, ax = _get_context(current_figure, current_ax, figure_size)
    figure.set_dpi(dpi if dpi is not None else config.figure_dpi)
    return _plot1d(current_ax=ax, current_figure=figure, *args, **kwargs)


def plot_empty(
    *args,
    current_figure: Figure | None = None,
    current_ax: SpectrumAxes | None = None,
    dpi: Number = None,
    figure_size: tuple[Number, Number] | tuple[str, str] = None,
    **kwargs,
):
    """
    Wrapper for empty 2D spectrum plotting using MiraPlot.
    """
    figure, ax = _get_context(current_figure, current_ax, figure_size)
    figure.set_dpi(dpi if dpi is not None else config.figure_dpi)

    plt.text(0.4, 0.5, "No spectrum available.")
    return figure, ax


# ----------------------------------------------------------------------------------------------------------------------
# Utility functions to set background colors
#


def set_figure_background(
    color: str = "white",
    current_figure: Figure | None = None,
    current_ax: SpectrumAxes | None = None,
    **kwargs,
):
    """
    Set the background color of the figure canvas.
    """
    figure, ax = _get_context(current_figure, current_ax, spectrum_axes=False)
    figure.patch.set_facecolor(Color(color).hex)


def set_axes_background(
    color: str | list[str] = "white",
    current_figure: Figure | None = None,
    current_ax: SpectrumAxes | None = None,
    **kwargs,
):
    """
    Set the background color for each axis in the figure. Accepts a list for multiple axes.
    """
    if isinstance(color, Color):
        color = [color]

    figure, ax = _get_context(current_figure, current_ax, spectrum_axes=False)
    axes = figure.get_axes()

    for i, ax in enumerate(axes):
        ax.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                color=Color(color[i % len(color)]).hex,
                zorder=-1,
            )
        )  # Ensure it's behind content


# ----------------------------------------------------------------------------------------------------------------------
# Wrapper functions for matplotlib
#


def figure(*args, **kwargs):
    """Create a new matplotlib figure."""
    return plt.figure(*args, **kwargs)


def show(*args, **kwargs):
    """Display the plot."""
    plt.show(*args, **kwargs)


def tight_layout(*args, **kwargs):
    """Adjust subplot parameters to fit into figure area."""
    plt.tight_layout(*args, **kwargs)


def savefig(*args, **kwargs):
    """Save the current figure to a file."""
    plt.savefig(*args, **kwargs)


def close(*args, **kwargs):
    """Close the current figure."""
    plt.close(*args, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
