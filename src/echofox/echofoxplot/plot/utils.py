import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypedDict

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


def add_res_labels(
    axs: Axes | Iterable[Axes],
    seq: str,
    *,
    seq_offset: int = 0,
    xticklabel_kwargs: dict | None = None,
) -> None:
    """Add residue labels to the x-axis of one or more Matplotlib axes.

    The x-axis ticks are set to residue positions and labeled with the
    one-letter amino acid code plus residue number, for example ``A1``,
    ``G2``, ``S3``. ``seq_offset`` can be used when the sequence does not
    start at residue 1.

    Parameters
    ----------
    axs : Axes or Iterable[Axes]
        Single Matplotlib axis or iterable of axes to label.
    seq : str
        Amin acid sequence used to generate residue labels.
    seq_offset : int, optional
        Offset added to residue numbering. For example, ``seq_offset=10``
        labels the first residue as position 11.
    xticklabel_kwargs : dict, optional
        Keyword arguments passed to ``ax.set_xticklabels`` for formatting the
        x-axis tick labels. If omitted, labels are rotated vertically and
        shown with the default Matplotlib x-tick label size.

    Returns
    -------
    None
        The axes are modified in place.
    """
    if xticklabel_kwargs is None:
        xticklabel_kwargs = {
            "rotation": 90,
            "fontsize": plt.rcParams["xtick.labelsize"],
        }

    res_full_ids = [f"{seq[i]}{i + 1 + seq_offset}" for i in range(len(seq))]
    ticks = np.arange(seq_offset + 1, seq_offset + len(seq) + 1)

    for ax in flatten_axs_list(axs):
        ax.set_xlim(
            seq_offset + 1 - 0.5,
            seq_offset + len(seq) + 1 - 0.5,
        )
        ax.set_xticks(ticks)
        ax.set_xticklabels(res_full_ids, **xticklabel_kwargs)


def highlight_res_labels(
    axs: Axes | Iterable[Axes],
    res_list: str | re.Pattern[str] | Iterable[str | re.Pattern[str]],
    *,
    label_styles: Mapping[str, Any] | None = None,
) -> None:
    """
    Highlight selected residue labels on the x-axis of one or more Matplotlib axes.

    Residue labels are matched against the existing x-axis tick-label text. Items in
    ``res_list`` are interpreted as regular expressions when possible. If a string is
    not a valid regular expression, it is treated as a literal label.

    Parameters
    ----------
    axs : Axes or Iterable[Axes]
        Single Matplotlib axis or iterable of axes whose x-axis labels should be
        modified.
    res_list : str, re.Pattern, or Iterable[str | re.Pattern]
        Residue label pattern or patterns to highlight. Examples include ``"A12"``,
        ``"A12|G13"``, ``"[ST]\\d+"``, or a compiled regular expression.
    label_styles : Mapping[str, Any], optional
        Matplotlib text properties applied to matching tick labels. For example:
        ``{"weight": "bold", "color": "red", "fontsize": 12}``.

        By default, matching labels are made bold.

    Returns
    -------
    None
        The axes are modified in place.
    """
    if label_styles is None:
        label_styles = {"weight": "bold"}

    if isinstance(res_list, str | re.Pattern):
        res_list = [res_list]

    compiled_patterns: list[re.Pattern[str]] = []

    for item in res_list:
        if isinstance(item, re.Pattern):
            compiled_patterns.append(item)
            continue

        try:
            compiled_patterns.append(re.compile(item))
        except re.error:
            compiled_patterns.append(re.compile(re.escape(item)))

    for ax in flatten_axs_list(axs):
        for label in ax.get_xticklabels():
            if any(pattern.search(label.get_text()) for pattern in compiled_patterns):
                label.set(**label_styles)
