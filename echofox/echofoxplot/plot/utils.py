from collections.abc import Iterable, Sequence
from typing import Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from echofox.core.colors import Color
from echofox.nmr.spectrum import NmrSpectrum


def _add_label(ax, label_list, label_kwargs_list) -> None:
    for i, label in enumerate(label_list):
        label_kwargs = label_kwargs_list[i]

        # add default kwargs for label to label_kwargs
        label_kwargs.setdefault("transform", ax.transAxes)
        label_kwargs.setdefault("fontsize", plt.rcParams["axes.labelsize"])
        label_kwargs.setdefault("verticalalignment", "top")
        label_kwargs.setdefault("horizontalalignment", "left")
        label_kwargs.setdefault("label_position", (0.02, 0.98))

        if label_kwargs["label_position"] in ["top-left", "top-right"]:
            padding = 5

            # Convert padding from points to axes coordinates
            pad_x = padding / ax.figure.dpi  # Convert pixels to figure-relative units
            pad_y = padding / ax.figure.dpi

            # Compute the top-left corner position (adjusted for padding)
            ax_x_min, ax_x_max = ax.get_xlim()
            ax_y_min, ax_y_max = ax.get_ylim()

            if label_kwargs["label_position"] == "top-left":
                # Adjust position to ensure the text fits inside the axes
                text_x = ax_x_min + pad_x * (ax_x_max - ax_x_min)
                text_y = ax_y_max - pad_y * (ax_y_max - ax_y_min)
                label_horizontalalignment = "left"

            elif label_kwargs["label_position"] == "top-right":
                # Adjust position to ensure the text fits inside the axes
                text_x = ax_x_max - pad_x * (ax_x_max - ax_x_min)
                text_y = ax_y_max - pad_y * (ax_y_max - ax_y_min)
                label_horizontalalignment = "right"
            else:
                text_x, text_y = label_kwargs["label_position"]
                label_horizontalalignment = "left"

            label_kwargs["horizontalalignment"] = label_horizontalalignment
            label_kwargs.pop("transform")

        else:
            text_x, text_y = label_kwargs["label_position"]

        # remove non-matplotlib kwargs from dict
        label_kwargs.pop("label_position")

        ax.text(text_x, text_y, label, **label_kwargs)


def _add_legend(ax, legend_kwargs) -> None:
    custom_lines = []
    custom_labels = []
    for i, spectrum in enumerate(ax.spectra):
        if isinstance(spectrum[0], NmrSpectrum) and spectrum[0].ndim == 1:
            dimensionality = 1
        elif isinstance(spectrum[0], NmrSpectrum) and spectrum[0].ndim == 2:
            dimensionality = 2
        else:
            raise ValueError("Spectrum must be an NmrSpectrum object.")

        name = spectrum[0].name
        spectrum_properties = spectrum[1]
        if dimensionality == 2:
            if "contour_color_positive" in spectrum_properties.keys():
                color = spectrum_properties["contour_color_positive"]
            elif "contour_color_negative" in spectrum_properties.keys():
                color = spectrum_properties["contour_color_negative"]
            else:
                color = ax.cmap_positive[i % len(ax.cmap_positive)]

        elif dimensionality == 1:
            if "spectrum_color" in spectrum_properties.keys():
                color = spectrum_properties["spectrum_color"]
            else:
                color = ax.color1d[i % len(ax.color1d)]
        else:
            color = "black"

        custom_lines.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2,
            )
        )
        custom_labels.append(name)

    if "loc" not in legend_kwargs.keys():
        legend_kwargs["loc"] = "upper left"
    if "frameon" not in legend_kwargs.keys():
        legend_kwargs["frameon"] = False
    if "handlelength" not in legend_kwargs.keys():
        legend_kwargs["handlelength"] = 1

    ax.legend(custom_lines, custom_labels, **legend_kwargs)


def configure_font(
    font: str = "Helvetica",
    family: Literal["serif", "sans-serif"] = "sans-serif",
    sizes: tuple[int, int, int] = (8, 10, 12),
) -> None:
    """
    Configure Matplotlib font settings for editable PDF/PS output.

    Parameters
    ----------
    font
        Name of the sans-serif font to use, e.g. "Helvetica", "Arial",
        or "DejaVu Sans". The font must be installed and discoverable by
        Matplotlib.
    sizes
        Three font sizes: default text size, axes/tick/legend size,
        and figure title size.
    """
    small, medium, large = sizes

    rcParams["pdf.fonttype"] = 42  # TrueType fonts
    rcParams["ps.fonttype"] = 42  # PostScript compatibility

    rcParams["font.family"] = family

    if family == "serif":
        rcParams["font.serif"] = [font, "Times New Roman", "DejaVu Serif"]
    else:
        rcParams["font.sans-serif"] = [font, "Helvetica", "Arial", "DejaVu Sans"]

    plt.rc("font", size=small)  # controls default text sizes
    plt.rc("axes", titlesize=medium, labelsize=medium)  # axes title and axis label sizes
    plt.rc("xtick", labelsize=small)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small)  # legend fontsize
    plt.rc("figure", titlesize=large)  # fontsize of the figure title


class LegendLabel(TypedDict):
    """One legend entry: the text to show + the color to apply to that text."""

    text: str
    color: str | Color


def fancy_legend(
    ax: Axes, labels: Sequence[LegendLabel], loc: str = "upper left", frameon: bool = False, **kwargs
) -> Legend:
    # Create dummy artists per legend label
    for _ in range(len(labels)):
        ax.plot([], [], " ", label="")

    legend = ax.legend(
        [label["text"] for label in labels], loc=loc, frameon=frameon, handlelength=0, handletextpad=0, **kwargs
    )
    legend.set_zorder(9999)

    # Color legend labels
    for i, line in enumerate(labels):
        plt.setp(legend.get_texts()[i], color=line["color"])

    if legend is None:
        raise ValueError("add_custom_legend() expected at least one Axes, got none.")

    return legend


def flatten_axs_list(axs: Axes | Iterable[Axes]) -> list[Axes]:
    """Normalize a single Axes or iterable/array of Axes to a flat list."""
    if isinstance(axs, Axes):
        return [axs]

    if isinstance(axs, np.ndarray):
        return list(axs.ravel())

    return list(axs)