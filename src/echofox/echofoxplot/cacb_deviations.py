import pandas as pd
from matplotlib.pyplot import rcParams

from ..nmr.cacb_deviations import get_cacb_deviations
from ..utils.greek_letters import GreekLetters
from .echofoxplot import bar_unassigned_residues


def plot_cacb_deviations(
    ax,
    df_random_coil: pd.DataFrame | None = None,
    df_cacb: pd.DataFrame | list[pd.DataFrame] | None = None,
    *,
    cacb_deviations: pd.DataFrame | None = None,
    atom_column: str | None = None,
    index_column: str | None = None,
    chemical_shift_column: str | None = None,
    colors: str | list[str] | None = None,
    bar_kwargs: dict | None = None,
    bar_missing: bool = True,
    unassigned_vspan_kwargs: dict | None = None,
    bar_zero: bool = True,
    bar_zero_kwargs: dict | None = None,
    ylim: tuple | None = None,
    xlim: tuple | None = None,
    ylabel: str | None = None,
    xlabel: str | None = "Residue Number",
    title: str | None = None,
    annotate_ylabel: bool = True,
):
    """
    Plot CA-CB chemical-shift deviations as a residue-wise bar plot.

    The function can either calculate CA-CB deviations internally from
    random-coil and observed chemical-shift DataFrames, or plot precomputed
    deviations supplied via `cacb_deviations`.

    Parameters
    ----------
    ax:
        Matplotlib axes object on which the plot is drawn.

    df_random_coil:
        DataFrame containing random-coil reference CA and CB chemical shifts.
        Required only if `cacb_deviations` is not supplied.

    df_cacb:
        DataFrame, or list of DataFrames, containing observed CA/CB chemical
        shifts. Required only if `cacb_deviations` is not supplied.

    cacb_deviations:
        Optional precomputed DataFrame containing `index_column` and
        `"cacb_deviation"` columns. If supplied, no deviations are calculated
        internally.

    atom_column:
        Column identifying atom names in `df_cacb`. Defaults to `"atom_3"`.

    index_column:
        Residue-index column. Defaults to `"residue_index"`.

    chemical_shift_column:
        Chemical-shift column in `df_cacb`. Defaults to `"chemical_shift_3"`.

    colors:
        Bar color specification. If a single color string is passed, all bars
        use that color. If a list of two colors is passed, the first color is
        used for positive deviations and the second for negative deviations.

    bar_kwargs:
        Additional keyword arguments passed to `ax.bar`.

    bar_missing:
        If True, mark unassigned or missing residues using
        `bar_unassigned_residues`.

    unassigned_vspan_kwargs:
        Keyword arguments passed to `bar_unassigned_residues`.

    bar_zero:
        If True, draw a horizontal zero line.

    bar_zero_kwargs:
        Keyword arguments used for the zero line.

    ylim:
        Optional y-axis limits.

    xlim:
        Optional x-axis limits.

    ylabel:
        Optional y-axis label. If None, a CA-CB deviation label is used.

    xlabel:
        Optional x-axis label. If None, no x-axis label is set.

    title:
        Optional plot title.

    annotate_ylabel:
        If True, annotate the y-axis with alpha-helix and beta-sheet labels.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes object.
    """

    if atom_column is None:
        atom_column = "atom_3"

    if index_column is None:
        index_column = "residue_index"

    if chemical_shift_column is None:
        chemical_shift_column = "chemical_shift_3"

    if bar_kwargs is None:
        bar_kwargs = {
            "width": 1.05,
            "linewidth": 0.0,
            "edgecolor": "black",
        }

    if bar_zero_kwargs is None:
        bar_zero_kwargs = {
            "linewidth": 1.0,
            "edgecolor": "black",
        }

    if cacb_deviations is None:
        if df_random_coil is None or df_cacb is None:
            raise ValueError(
                "Either pass `cacb_deviations`, or pass both "
                "`df_random_coil` and `df_cacb` so deviations can be calculated."
            )

        cacb_deviations = get_cacb_deviations(
            df_random_coil=df_random_coil,
            df_cacb=df_cacb,
            atom_column=atom_column,
            index_column=index_column,
            chemical_shift_column=chemical_shift_column,
        )

    if ylabel is None:
        ylabel = r"Δδ($^{13}$C$_\text{α}$)" "\n" r"-Δδ($^{13}$C$_\text{β}$)"
    ax.set_ylabel(ylabel, rotation=0, ha="right", va="center")

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    cacb_deviations = cacb_deviations.sort_values(by=index_column)

    bar_colors = colors
    if isinstance(colors, list):
        if len(colors) != 2:
            raise ValueError(
                "`colors` must be either a single color string or a list of two colors: "
                "[positive_color, negative_color]."
            )

        positive_color, negative_color = colors

        bar_colors = [positive_color if value >= 0 else negative_color for value in cacb_deviations["cacb_deviation"]]

    ax.bar(
        cacb_deviations[index_column],
        cacb_deviations["cacb_deviation"],
        color=bar_colors,
        **bar_kwargs,
    )

    if bar_missing:
        bar_unassigned_residues(ax, vspan_kwargs=unassigned_vspan_kwargs)

    if bar_zero:
        ax.axhspan(
            ymin=0.0,
            ymax=0.0,
            xmin=0.0,
            xmax=1.0,
            **bar_zero_kwargs,
        )

    if annotate_ylabel:
        label_fontsize = rcParams["axes.labelsize"]

        ax.figure.canvas.draw()
        ylabel = ax.yaxis.get_label()
        ylabel_x = ylabel.get_position()[0]
        ylabel_transform = ylabel.get_transform()

        for text, y, va in [
            (f"{GreekLetters.alpha}-Helix", 1.0, "top"),
            (f"{GreekLetters.beta}-Sheet", 0.0, "bottom"),
        ]:
            ax.text(
                ylabel_x,
                y,
                text,
                transform=ylabel_transform,
                ha="right",
                va=va,
                fontsize=label_fontsize,
                clip_on=False,
            )

    if ylim:
        ax.set_ylim(ylim)

    if xlim:
        ax.set_xlim(xlim)

    if title:
        ax.set_title(title)

    return ax
