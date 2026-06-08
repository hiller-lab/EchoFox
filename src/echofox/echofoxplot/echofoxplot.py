import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Import echofox.core components
from echofox.core.colors import Color
from echofox.core.typing import Number

# Import echofoxplot axes
from echofox.echofoxplot.axes.spectrum_axes import SpectrumAxes
from echofox.echofoxplot.plot.spectrum.spectrum1d import plot1d as _plot1d

# Import plotting functions from echofoxplot
from echofox.echofoxplot.plot.spectrum.spectrum2d import plot2d as _plot2d
from echofox.echofoxplot.plot.utils import flatten_axs_list
from echofox.utils.greek_letters import GreekLetters
from echofox.utils.units import convert_to_inches

# Import echofoxplot config
from .config import config

# Configure font types for PDF and PostScript output to ensure compatibility with Illustrator and others
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["figure.constrained_layout.use"] = True

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
                ax = figure.add_subplot(111, projection="SpectrumAxes")  # Custom projection
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
    return _plot2d(*args, current_ax=ax, current_figure=figure, **kwargs)


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
    return _plot1d(*args, current_ax=ax, current_figure=figure, **kwargs)


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


def plot_csps(
    ax,
    df_ref: pd.DataFrame,
    df_comp: pd.DataFrame,
    chemical_shift_columns: list[str] | None = None,
    weight: float = 1 / 5,
    color_bars: str = "darkgray",
    color_significant: str = "#f79132",
    bar_kwargs: dict | None = None,
    plot_average: bool = True,
    avg_hline_kwargs: dict | None = None,
    nr_sigmas: int = 1,
    sigma_hline_kwargs: dict | None = None,
    draw_legend: bool = True,
    legend_kwargs: dict | None = None,
    print_significant: bool | str = False,
    bar_unassigned: bool = True,
    unassigned_vspan_kwargs: dict | None = None,
    ylim: tuple | None = None,
    ylabel: str | None = None,
    xlabel: str | None = "Residue Number",
    title: str | None = None,
):
    """
    Plot residue-wise chemical shift perturbations (CSPs) between two data frames.

    The function matches residues in `df_comp` to residues in `df_ref` using the
    `residue_index` column, calculates a weighted two-dimensional CSP for each
    matched residue, and plots the result as a bar plot. Bars above the significance
    threshold, defined as the mean CSP plus `nr_sigmas` standard deviations, are
    colored separately. Optional horizontal lines indicate the mean CSP and the
    significance threshold.

    The CSP is calculated as:

        sqrt((cs_comp_1 - cs_ref_1)^2 + ((cs_comp_2 - cs_ref_2) * weight)^2)

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object on which the CSP bar plot is drawn.
    df_ref : pandas.DataFrame
        Reference data frame containing residue indices and chemical shift columns.
    df_comp : pandas.DataFrame
        Comparison data frame containing residue indices and chemical shift columns.
    chemical_shift_columns : list[str] | None, optional
        Names of the two chemical shift columns used for CSP calculation.
        If None, defaults to ["chemical_shift_1", "chemical_shift_2"].
    weight : float, optional
        Scaling factor applied to the second chemical shift dimension.
        Default is 1/5.
    color_bars : str, optional
        Color used for bars below or equal to the significance threshold.
    color_significant : str, optional
        Color used for bars above the significance threshold and for default
        threshold lines.
    bar_kwargs : dict | None, optional
        Additional keyword arguments passed to `ax.bar`. If None, sensible defaults
        are used.
    plot_average : bool, optional
        If True, draw the average CSP line and, if `nr_sigmas > 0`, the threshold
        line.
    avg_hline_kwargs : dict | None, optional
        Keyword arguments passed to `ax.hlines` for the average CSP line.
    sigma_hline_kwargs : dict | None, optional
        Keyword arguments passed to `ax.hlines` for the significance threshold line.
    nr_sigmas : int, optional
        Number of standard deviations added to the mean CSP to define the
        significance threshold. If 0 or lower, no sigma threshold line is drawn.
    draw_legend : bool, optional
        If True and `plot_average` is True, draw a legend.
    legend_kwargs : dict | None, optional
        Keyword arguments passed to `ax.legend`.
    print_significant : bool | str, optional
        If True or "pymol", print a PyMOL residue selection for significant
        residues. If "blender", print significant residue numbers as a
        comma-separated list.
    bar_unassigned : bool, optional
        If True, call `bar_unassigned_residues` to visually mark residues without
        plotted data.
    unassigned_vspan_kwargs : dict | None, optional
        Keyword arguments passed to `bar_unassigned_residues` for styling the
        unassigned-residue spans.
    ylim : tuple | None, optional
        Y-axis limits passed to `ax.set_ylim`.
    ylabel : str | None, optional
        Y-axis label. If None, a default CSP label is used.
    xlabel : str | None, optional
        X-axis label. If None, no x-axis label is set.
    title : str | None, optional
        Plot title. If None, no title is set.

    Returns
    -------
    matplotlib.axes.Axes
        The input axes object containing the CSP plot.
    """
    if chemical_shift_columns is None:
        chemical_shift_columns = ["chemical_shift_1", "chemical_shift_2"]

    if bar_kwargs is None:
        bar_kwargs = {
            "width": 1.05,
            "linewidth": 0.0,
            "edgecolor": "black",
        }

    if avg_hline_kwargs is None:
        avg_hline_kwargs = {
            "color": color_significant,
            "linestyle": "-",
            "label": "Avg.",
            "lw": 1.0,
            "alpha": 0.9,
        }

    if sigma_hline_kwargs is None:
        sigma_hline_kwargs = {
            "color": color_significant,
            "linestyle": "--",
            "label": f"{nr_sigmas}{GreekLetters.sigma}",
            "lw": 1.0,
            "alpha": 0.8,
        }

    if legend_kwargs is None:
        legend_kwargs = {
            "loc": "lower left",
            "bbox_to_anchor": (0.0, 0.95),
            "ncol": 3,
            "frameon": False,
        }

    if ylabel is None:
        ylabel = f"{GreekLetters.Delta}{GreekLetters.delta}HN\n[ppm]"
    ax.set_ylabel(ylabel, rotation=0, ha="right")

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    df_ref = df_ref.reset_index()
    df_comp = df_comp.reset_index()

    x = []
    y = []
    for row_index in range(len(df_comp)):
        res_comp_idx = df_comp["residue_index"][row_index]
        pos_comp = (df_comp[chemical_shift_columns[0]][row_index], df_comp[chemical_shift_columns[1]][row_index])

        row_ref = df_ref[df_ref["residue_index"] == res_comp_idx]
        if row_ref.empty:
            continue

        pos_ref = (row_ref[chemical_shift_columns[0]].to_list()[0], row_ref[chemical_shift_columns[1]].to_list()[0])

        x.append(int(row_ref["residue_index"].to_list()[0]))
        y.append(np.sqrt((pos_comp[0] - pos_ref[0]) ** 2 + ((pos_comp[1] - pos_ref[1]) * weight) ** 2))

    x = np.array(x)
    y = np.array(y)

    y_avg = np.average(y)
    y_std = np.std(y)
    threshold = y_avg + y_std * nr_sigmas

    colors = [{v <= threshold: color_bars, threshold < v: color_significant}[True] for v in y]

    if print_significant or print_significant == "pymol":
        print("select sign, resi " + "+".join([str(i) for i, v in zip(x, y) if threshold < v]))
    elif print_significant == "blender":
        print(",".join([str(i) for i, v in zip(x, y) if threshold < v]))

    if plot_average:
        ax.hlines(y_avg, *ax.get_xlim(), **avg_hline_kwargs)

        if nr_sigmas > 0:
            ax.hlines(y_avg + y_std * nr_sigmas, *ax.get_xlim(), **sigma_hline_kwargs)

    ax.bar(x, y, color=colors, **bar_kwargs)

    if bar_unassigned:
        bar_unassigned_residues(ax, vspan_kwargs=unassigned_vspan_kwargs)

    if draw_legend and plot_average:
        ax.legend(**legend_kwargs)

    if ylim:
        ax.set_ylim(ylim)

    if title:
        ax.set_title(title)

    return ax


def get_irs(
    df_ref: pd.DataFrame,
    df_comp: pd.DataFrame,
):
    """
    Calculate residue-wise intensity ratios between two data frames.

    This is a convenience wrapper around `get_property_ratios` using the
    `"intensity"` column. For each residue present in both `df_ref` and `df_comp`,
    the function calculates the ratio:

        intensity_comp / intensity_ref

    Only residues with positive, non-missing values in both data frames are included.

    Parameters
    ----------
    df_ref : pandas.DataFrame
        Reference data frame containing residue indices and an intensity column.
    df_comp : pandas.DataFrame
        Comparison data frame containing residue indices and an intensity column.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float, float]
        A tuple containing the residue indices, intensity ratios, average ratio,
        and standard deviation of the ratios.
    """
    return get_property_ratios("intensity", df_ref, df_comp)


def get_vrs(
    df_ref: pd.DataFrame,
    df_comp: pd.DataFrame,
):
    """
    Calculate residue-wise volume ratios between two data frames.

    This is a convenience wrapper around `get_property_ratios` using the
    `"volume"` column. For each residue present in both `df_ref` and `df_comp`,
    the function calculates the ratio:

        volume_comp / volume_ref

    Only residues with positive, non-missing values in both data frames are included.

    Parameters
    ----------
    df_ref : pandas.DataFrame
        Reference data frame containing residue indices and a volume column.
    df_comp : pandas.DataFrame
        Comparison data frame containing residue indices and a volume column.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float, float]
        A tuple containing the residue indices, volume ratios, average ratio,
        and standard deviation of the ratios.
    """
    return get_property_ratios("volume", df_ref, df_comp)


def get_property_ratios(
    property: str,
    df_ref: pd.DataFrame,
    df_comp: pd.DataFrame,
    res_index_col_name: str = "residue_index",
):
    """
    Calculate residue-wise ratios for a numeric property between two data frames.

    The function matches residues in `df_comp` to residues in `df_ref` using
    `res_index_col_name`, extracts the selected property from both data frames,
    and calculates the ratio:

        value_comp / value_ref

    Only residues present in both data frames are included. Missing values,
    non-positive reference values, and non-positive comparison values are ignored.

    Parameters
    ----------
    property : str
        Name of the numeric column for which ratios should be calculated.
    df_ref : pandas.DataFrame
        Reference data frame containing residue indices and the selected property.
    df_comp : pandas.DataFrame
        Comparison data frame containing residue indices and the selected property.
    res_index_col_name : str, optional
        Name of the column containing residue indices. Default is "residue_index".

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float, float]
        A tuple containing:

        - residue indices as an array of integers,
        - property ratios as an array of floats,
        - average property ratio,
        - standard deviation of the property ratios.

    Notes
    -----
    The function assumes that residue indices can be converted to integers.
    If multiple rows with the same residue index are present in `df_ref`, the first
    matching row is used.
    """
    df_ref = df_ref.reset_index()
    df_comp = df_comp.reset_index()

    x = []
    y = []
    for row_index in range(len(df_comp)):
        res_comp_idx = df_comp[res_index_col_name][row_index]
        val_comp = df_comp[property][row_index]

        row_ref = df_ref[df_ref[res_index_col_name] == res_comp_idx]
        if row_ref.empty:
            continue

        val_ref = row_ref[property].iloc[0]

        if pd.notna(val_ref) and pd.notna(val_comp) and val_ref > 0 and val_comp > 0:
            x.append(int(row_ref[res_index_col_name].iloc[0]))
            y.append(val_comp / val_ref)

    x = np.array(x)
    y = np.array(y)
    y_avg = np.average(y)
    y_std = np.std(y)

    return x, y, y_avg, y_std


def plot_irs(
    ax,
    df_ref: pd.DataFrame,
    df_comp: pd.DataFrame,
    intensity_column: str | None = None,
    color_bars: str = "darkgray",
    color_significant: str = "#f79132",
    bar_kwargs: dict | None = None,
    plot_average: bool = True,
    avg_hline_kwargs: dict | None = None,
    nr_sigmas: int = 1,
    sigma_hline_kwargs: dict | None = None,
    draw_legend: bool = True,
    legend_kwargs: dict | None = None,
    print_significant: bool = False,
    bar_unassigned: bool = True,
    unassigned_vspan_kwargs: dict | None = None,
    bar_one: bool = False,
    bar_one_kwargs: dict | None = None,
    only_peaks_with_volume: bool = False,
    lower_irs_only: bool = False,
    ignore_outliers_over: float | int | None = None,
    ylim: tuple | None = None,
    ylabel: str | None = None,
    xlabel: str | None = "Residue Number",
    title: str | None = None,
):
    """
    Plot residue-wise intensity ratios between two data frames.

    The function calculates intensity ratios between `df_comp` and `df_ref` using
    `get_irs` and visualizes them as a residue-wise bar plot. Bars can be colored
    according to whether they deviate from the mean ratio by more than
    `nr_sigmas` standard deviations. Optional horizontal lines indicate the mean
    ratio and the corresponding sigma threshold or thresholds.

    By default, the plotted ratio is:

        I / I0 = intensity_comp / intensity_ref

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object on which the intensity-ratio bar plot is drawn.
    df_ref : pandas.DataFrame
        Reference data frame containing residue indices and intensity values.
    df_comp : pandas.DataFrame
        Comparison data frame containing residue indices and intensity values.
    intensity_column : str | None, optional
        Name of the intensity column. If None, defaults to "intensity".
        Currently, ratio calculation is delegated to `get_irs`, which uses the
        "intensity" column.
    color_bars : str, optional
        Color used for bars that are not classified as significant.
    color_significant : str, optional
        Color used for bars classified as significant and for the default average
        and sigma-threshold lines.
    bar_kwargs : dict | None, optional
        Additional keyword arguments passed to `ax.bar`. If None, sensible defaults
        are used.
    plot_average : bool, optional
        If True, draw the average intensity-ratio line and, if `nr_sigmas > 0`,
        sigma-threshold lines.
    avg_hline_kwargs : dict | None, optional
        Keyword arguments passed to `ax.hlines` for the average ratio line.
    sigma_hline_kwargs : dict | None, optional
        Keyword arguments passed to `ax.hlines` for the sigma-threshold line or
        lines.
    nr_sigmas : int, optional
        Number of standard deviations used to define significant deviations from
        the mean ratio.
    draw_legend : bool, optional
        If True and `plot_average` is True, draw a legend.
    legend_kwargs : dict | None, optional
        Keyword arguments passed to `ax.legend`.
    print_significant : bool, optional
        If True or "pymol", print a PyMOL residue selection for significant
        residues. If "blender", print significant residue numbers as a
        comma-separated list.
    bar_unassigned : bool, optional
        If True, call `bar_unassigned_residues` to visually mark residue positions
        without plotted data.
    unassigned_vspan_kwargs : dict | None, optional
        Keyword arguments passed to `bar_unassigned_residues` for styling the
        unassigned-residue spans.
    bar_one : bool, optional
        If True, draw a horizontal reference marker at ratio 1.0.
    bar_one_kwargs : dict | None, optional
        Keyword arguments used to style the ratio-1.0 reference marker.
    only_peaks_with_volume : bool, optional
        If True, restrict both data frames to rows with non-missing volume values
        before calculating intensity ratios.
    lower_irs_only : bool, optional
        If True, only ratios below `mean - nr_sigmas * std` are classified as
        significant. If False, both high and low deviations from the mean are
        classified as significant.
    ignore_outliers_over : float | int | None, optional
        If given, remove ratios larger than this value before calculating the mean,
        standard deviation, and plot.
    ylim : tuple | None, optional
        Y-axis limits passed to `ax.set_ylim`.
    ylabel : str | None, optional
        Y-axis label. If None, defaults to an intensity-ratio label.
    xlabel : str | None, optional
        X-axis label. If None, no x-axis label is set.
    title : str | None, optional
        Plot title. If None, no title is set.

    Returns
    -------
    matplotlib.axes.Axes
        The input axes object containing the intensity-ratio plot.
    """
    if intensity_column is None:
        intensity_column = "intensity"

    if bar_kwargs is None:
        bar_kwargs = {
            "width": 1.05,
            "linewidth": 0.0,
            "edgecolor": "black",
        }

    if avg_hline_kwargs is None:
        avg_hline_kwargs = {
            "color": color_significant,
            "linestyle": "-",
            "label": "Avg.",
            "lw": 1.0,
            "alpha": 0.9,
        }

    if sigma_hline_kwargs is None:
        sigma_hline_kwargs = {
            "color": color_significant,
            "linestyle": "--",
            "label": f"{nr_sigmas}{GreekLetters.sigma}",
            "lw": 1.0,
            "alpha": 0.8,
        }

    if legend_kwargs is None:
        legend_kwargs = {
            "loc": "lower left",
            "bbox_to_anchor": (0.0, 0.95),
            "ncol": 3,
            "frameon": False,
        }

    if bar_one_kwargs is None:
        bar_one_kwargs = {
            "linewidth": 1.0,
            "edgecolor": "black",
        }

    if ylabel is None:
        ylabel = r"I/I$_{0}$"
    ax.set_ylabel(ylabel, rotation=0, ha="right")

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if only_peaks_with_volume:
        df_ref = df_ref[df_ref["volume"].notna()]
        df_comp = df_comp[df_comp["volume"].notna()]

    x, y, y_avg, y_std = get_irs(df_ref, df_comp)

    if ignore_outliers_over is not None:
        mask = y <= ignore_outliers_over
        x = x[mask]
        y = y[mask]
        y_avg = np.average(y)
        y_std = np.std(y)

    if not lower_irs_only:
        colors = [
            {
                np.abs(v - y_avg) > y_std * nr_sigmas: color_significant,
            }.get(True, color_bars)
            for v in y
        ]
    else:
        colors = [
            {
                v < y_avg - y_std * nr_sigmas: color_significant,
            }.get(True, color_bars)
            for v in y
        ]

    if print_significant or print_significant == "pymol":
        print("select sign, resi " + "+".join([str(i) for i, c in zip(x, colors) if c != color_bars]))
    elif print_significant == "blender":
        print(",".join([str(i) for i, c in zip(x, colors) if c != color_bars]))

    if plot_average:
        ax.hlines(y_avg, *ax.get_xlim(), **avg_hline_kwargs)

        if nr_sigmas > 0:
            ax.hlines(y_avg - y_std * nr_sigmas, *ax.get_xlim(), **sigma_hline_kwargs)
            if not lower_irs_only:
                ax.hlines(
                    y_avg + y_std * nr_sigmas,
                    *ax.get_xlim(),
                    **{**sigma_hline_kwargs, "label": "_nolegend_"},
                )

    ax.bar(x, y, color=colors, **bar_kwargs)

    if bar_one:
        ax.axhspan(
            ymin=1.0,
            ymax=1.0,
            xmin=0.0,
            xmax=1.0,
            **bar_one_kwargs,
        )

    if bar_unassigned:
        bar_unassigned_residues(ax, vspan_kwargs=unassigned_vspan_kwargs)

    if draw_legend and plot_average:
        ax.legend(**legend_kwargs)

    if ylim:
        ax.set_ylim(ylim)

    if title:
        ax.set_title(title)

    return ax


def bar_unassigned_residues(
    axs,
    x_range=None,
    vspan_kwargs: dict | None = None,
) -> None:
    """
    Shade residue positions that do not contain plotted data on one or more axes.

    The function inspects existing plot elements on each axes object, including
    lines, bar containers, and scatter collections, to identify x-positions that
    already contain data. Positions within `x_range` that do not overlap with any
    detected data x-position are considered unassigned and are shaded using
    `bar_residues`.

    Parameters
    ----------
    axs : matplotlib.axes.Axes or iterable of matplotlib.axes.Axes
        Axes object or collection of axes objects to inspect and annotate.
    x_range : iterable | None, optional
        Residue positions to check for missing data. If None, the current x-axis
        limits are used to generate an integer residue range.
    vspan_kwargs : dict | None, optional
        Keyword arguments passed to `bar_residues` and ultimately to `ax.axvspan`
        for styling the shaded regions.

    Returns
    -------
    None

    Notes
    -----
    The function determines occupied x-positions from the currently drawn artists.
    Therefore, it should be called after plotting the data that should be used to
    define assigned residue positions.
    """
    for ax in flatten_axs_list(axs):
        x_with_data = set()

        # Lines
        for line in ax.lines:
            x_with_data.update(line.get_xdata())

        # Bars
        for container in ax.containers:
            try:
                for patch in container:
                    if hasattr(patch, "get_x"):
                        # For bars: x is left edge; width gives span
                        x0 = patch.get_x()
                        w = patch.get_width()
                        # Add the center of bar OR full span
                        for x in np.linspace(x0, x0 + w, num=5):
                            x_with_data.add(x)
            except TypeError:
                pass

        # Scatter
        for col in ax.collections:
            if hasattr(col, "get_offsets"):
                offsets = col.get_offsets()
                if len(offsets) > 0:
                    xs = offsets[:, 0]
                    x_with_data.update(xs)

        if not x_with_data:
            return

        if x_range is None:
            xmin, xmax = ax.get_xlim()
            x_range = range(int(np.floor(xmin)), int(np.ceil(xmax)) + 1)

        x_with_data = np.array(list(x_with_data))

        empty_positions = [x for x in x_range if not np.any(np.isclose(x_with_data, x))]

        bar_residues(
            ax,
            empty_positions,
            vspan_kwargs,
        )

    return


def bar_residues(
    ax,
    indices,
    vspan_kwargs: dict | None = None,
):
    """
    Shade selected residue positions on an axes object.

    Each residue index in `indices` is highlighted by drawing a vertical span from
    `idx - 0.5` to `idx + 0.5`, covering the full height of the axes. This is useful
    for marking missing, unassigned, excluded, or otherwise highlighted residue
    positions in residue-wise plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object on which the residue spans are drawn.
    indices : iterable
        Residue indices to shade.
    vspan_kwargs : dict | None, optional
        Keyword arguments passed to `ax.axvspan`. If None, light-gray transparent
        spans without outlines are used.

    Returns
    -------
    None
    """
    if vspan_kwargs is None:
        vspan_kwargs = {
            "color": "lightgray",
            "alpha": 0.5,
            "lw": 0,
        }

    for idx in indices:
        # Shade the entire vertical range from y_range[0] to y_range[1]
        ax.axvspan(
            idx - 0.5,  # start shading half-step before residue center
            idx + 0.5,  # to half-step after
            ymin=0,
            ymax=1,  # full vertical range of the axes
            zorder=0,
            **vspan_kwargs,
        )


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


def subplots(
    *args,
    figsize: tuple[Number, Number] | tuple[str, str] | None = None,
    **kwargs,
):
    """
    Create a matplotlib figure and axes, with optional unit-aware figure sizing.

    This is a thin wrapper around ``matplotlib.pyplot.subplots``. All positional
    and keyword arguments are forwarded to ``plt.subplots`` unchanged, except for
    ``figsize``.

    ``figsize`` may be given as a tuple of numbers or strings. Numeric values are
    interpreted as inches, matching matplotlib's default behavior. String values
    may include a unit, for example ``"8 cm"`` or ``"12cm"``, and are converted
    to inches before the figure is created. If no unit is given, the value is
    interpreted as inches.

    Args:
        *args:
            Positional arguments passed to ``matplotlib.pyplot.subplots``.
        figsize:
            Optional figure size as ``(width, height)``. Values can be numbers
            in inches or strings such as ``"8 cm"``. If ``None``, matplotlib's
            default figure size is used.
        **kwargs:
            Additional keyword arguments passed to ``matplotlib.pyplot.subplots``.

    Returns:
        tuple:
            The ``(fig, ax)`` or ``(fig, axes)`` tuple returned by
            ``matplotlib.pyplot.subplots``.
    """
    return plt.subplots(
        *args, figsize=tuple(map(convert_to_inches, figsize)) if figsize is not None else None, **kwargs
    )


def show(*args, **kwargs) -> None:
    """Display the plot."""
    plt.show(*args, **kwargs)


def tight_layout(*args, **kwargs) -> None:
    """Adjust subplot parameters to fit into figure area."""
    plt.tight_layout(*args, **kwargs)


def savefig(*args, **kwargs) -> None:
    """Save the current figure to a file."""
    plt.savefig(*args, **kwargs)


def close(*args, **kwargs) -> None:
    """Close the current figure."""
    plt.close(*args, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
