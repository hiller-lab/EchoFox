import numpy as np

# matplotlib
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import StrMethodFormatter

# matplotlib.use('TkAgg')
rcParams["pdf.fonttype"] = 42  # Configure Matplotlib to use TrueType fonts
rcParams["ps.fonttype"] = 42  # If exporting to PostScript as well

from typing import Literal, TypeAlias

# echofox imports
from echofox.core.typing import Number
from echofox.echofoxplot.axes.spectrum_axes import SpectrumAxes
from echofox.echofoxplot.config import config
from echofox.echofoxplot.plot.spectrum.exceptions import (
    InvalidAxisError,
    InvalidInsetError,
    InvalidSpectrumIndexError,
    LabelMismatchError,
)
from echofox.echofoxplot.plot.utils import _add_label, _add_legend
from echofox.nmr.chemical_shift import ChemicalShift, PpmRange
from echofox.nmr.spectrum import NmrSpectrum, read_spectra

Spectra: TypeAlias = NmrSpectrum | list[NmrSpectrum]
TraceType2D: TypeAlias = list[
    ChemicalShift | Number,
    Literal["f1", "f2"],
    int | list[int] | Literal["all"],
    dict,
]
ProjectionType2D: TypeAlias = list[
    Literal["f1", "f2"], int | list[int] | Literal["all"], bool, dict
]


class Trace2D:
    def __init__(
        self,
        spectra: Spectra,
        chemical_shift: ChemicalShift | Number | str,
        axis: Literal["f1", "f2"],
        spectrum_index: int | list[int] = 0,
        trace_kwargs: dict = None,
    ):

        self.chemical_shift = chemical_shift
        self.axis = axis

        if not isinstance(spectra, list):
            spectra = [spectra]

        if spectrum_index is None:
            self.spectrum_index = np.arange(len(spectra))
        else:
            self.spectrum_index = spectrum_index

        if trace_kwargs is None:
            self.trace_kwargs = {}
        else:
            self.trace_kwargs = trace_kwargs

        self.data = []
        for i, spectrum in enumerate(spectra):
            if self.axis == "f2":
                if i in self.spectrum_index or self.spectrum_index == ["all"]:
                    self.data.append(spectrum.get_column(self.chemical_shift))
            elif self.axis == "f1":
                if i in self.spectrum_index or self.spectrum_index == ["all"]:
                    self.data.append(spectrum.get_row(self.chemical_shift))

    @property
    def spectrum_index(self):
        return self._spectrum_index

    @spectrum_index.setter
    def spectrum_index(self, val):
        if val == "all" or val == ["all"]:
            self._spectrum_index = ["all"]
        elif isinstance(val, int):  # single spec
            self._spectrum_index = np.asarray(val)
        elif not isinstance(val, list):
            raise InvalidSpectrumIndexError(val)
        else:
            self._spectrum_index = val

    @property
    def chemical_shift(self) -> ChemicalShift:
        return self._chemical_shift

    @chemical_shift.setter
    def chemical_shift(self, val: ChemicalShift | str | Number) -> None:
        if isinstance(val, ChemicalShift):
            self._chemical_shift = val
        else:
            self._chemical_shift = ChemicalShift(val)

    @property
    def axis(self) -> str:
        return self._axis

    @axis.setter
    def axis(self, val: Literal["f1", "f2"]) -> None:
        if val == "f1":
            self._axis = "f1"
        elif val == "f2":
            self._axis = "f2"
        else:
            raise InvalidAxisError(val)


class Projection2D:
    def __init__(
        self,
        spectra: Spectra,
        axis: Literal["f1", "f2"],
        spectrum_index: int | list[int] = 0,
        is_external: bool = True,
        projection_kwargs: dict = None,
    ):

        self.is_external = is_external
        self.axis = axis

        if not isinstance(spectra, list):
            spectra = [spectra]

        if spectrum_index is None:
            self.spectrum_index = ["all"]
        else:
            self.spectrum_index = spectrum_index

        if projection_kwargs is None:
            self.projection_kwargs = {}
        else:
            self.projection_kwargs = projection_kwargs

        self.data = []
        for i, spectrum in enumerate(spectra):
            if self.axis == "f2":
                if i in self.spectrum_index or self.spectrum_index == ["all"]:
                    self.data.append(spectrum.projection_f2)
            elif self.axis == "f1":
                if i in self.spectrum_index or self.spectrum_index == ["all"]:
                    self.data.append(spectrum.projection_f1)

    @property
    def spectrum_index(self):
        return self._spectrum_index

    @spectrum_index.setter
    def spectrum_index(self, val):
        if val == "all" or val == ["all"]:
            self._spectrum_index = ["all"]
        elif isinstance(val, int):  # single spec
            self._spectrum_index = np.asarray(val)
        elif not isinstance(val, list):
            raise InvalidSpectrumIndexError(val)
        else:
            self._spectrum_index = val

    @property
    def is_external(self) -> bool:
        return self._is_external

    @is_external.setter
    def is_external(
        self, val: Literal["projection_external", "projection_internal"] | bool
    ) -> None:
        if val == "projection_external":
            self._is_external = True
        elif val == "projection_internal":
            self._is_external = False
        else:
            self._is_external = val

    @property
    def axis(self) -> str:
        return self._axis

    @axis.setter
    def axis(self, val: Literal["f1", "f2"]) -> None:
        if val == "f1":
            self._axis = "f1"
        elif val == "f2":
            self._axis = "f2"
        else:
            raise InvalidAxisError(val)


def plot2d(
    spec,
    ppm_range: tuple[
        PpmRange | tuple[float | str | ChemicalShift, float | str | ChemicalShift],
        PpmRange | tuple[float | str | ChemicalShift, float | str | ChemicalShift],
    ] = (None, None),
    tick_spacing: list[
        list[Number, Number] | None, list[Number, Number] | None
    ] = config.tick_spacing_2d,
    label: str | None = None,
    label_kwargs=None,
    pick_peaks: bool = False,
    pick_peaks_kwargs: dict | None = None,
    expected_sidechainpeaks: int = 0,
    mark_peaks: bool = False,
    mark_peaks_kwargs: dict | None = None,
    show_f1_axis: bool = True,
    show_f2_axis: bool = True,
    show_f1_label: bool = True,
    show_f2_label: bool = True,
    f1_label_kwargs: dict | None = None,
    f2_label_kwargs: dict | None = None,
    f1_ticklabel_kwargs: dict | None = None,
    f2_ticklabel_kwargs: dict | None = None,
    current_figure: Figure | None = None,
    current_ax: SpectrumAxes | None = None,
    projection_f1_ratio: float = config.projection_f1_ratio,
    projection_f2_ratio: float = config.projection_f2_ratio,
    markers_f1=None,
    markers_f1_kwargs: dict | None = None,
    markers_f2=None,
    markers_f2_kwargs: dict | None = None,
    projections: ProjectionType2D | list[ProjectionType2D] | None = None,
    traces: TraceType2D | list[TraceType2D] | None = None,
    insets=None,
    legend: bool = False,
    legend_kwargs: dict | None = None,
    set_axis_off: bool = False,
    **kwargs,
):

    # set default lists and dicts
    if markers_f1 is None:
        markers_f1 = []
    if markers_f2 is None:
        markers_f2 = []
    if label_kwargs is None:
        label_kwargs = {}
    if f1_label_kwargs is None:
        f1_label_kwargs = {}
    if f2_label_kwargs is None:
        f2_label_kwargs = {}
    if f1_ticklabel_kwargs is None:
        f1_ticklabel_kwargs = {}
    if f2_ticklabel_kwargs is None:
        f2_ticklabel_kwargs = {}
    if legend_kwargs is None:
        legend_kwargs = {}
    if mark_peaks_kwargs is None:
        mark_peaks_kwargs = {}
    if pick_peaks_kwargs is None:
        pick_peaks_kwargs = {}

    # read spectra
    spectra = read_spectra(spec)

    # set current ax and figure
    figure = current_figure
    ax = current_ax

    # init variables
    f1_label_text, f2_label_text = None, None

    # plot spectrum and extract variables
    for i, spectrum in enumerate(spectra):
        new_dict = {}
        for key, value in kwargs.items():
            if isinstance(value, list):  # If the value is a list
                if len(value) > 1:
                    new_dict[key] = value[
                        i % len(value)
                    ]  # Get the element at index i (wrap around if needed)
                else:
                    new_dict[key] = value[0]  # If only one element, take it
            else:
                new_dict[key] = value  # If it's not a list, keep it as is

        ax.plot2d(spectrum, **new_dict)

        if f1_label_text is None:
            f1_label_text = spectrum.get_label_text(0)
        elif not f1_label_text == spectrum.get_label_text(0):
            raise LabelMismatchError("f1", f1_label_text, spectrum.get_label_text(0))

        if f2_label_text is None:
            f2_label_text = spectrum.get_label_text(1)
        elif not f2_label_text == spectrum.get_label_text(1):
            raise LabelMismatchError("f2", f2_label_text, spectrum.get_label_text(1))

        # set spectrum limits to max
        ax.set_xlim(
            spectrum.dimension_ranges[1].high_ppm, spectrum.dimension_ranges[1].low_ppm
        )
        ax.set_ylim(
            spectrum.dimension_ranges[0].high_ppm, spectrum.dimension_ranges[0].low_ppm
        )

    if ppm_range != (None, None):
        ppm_range = [PpmRange(*range) for range in ppm_range]
        # set spectrum limits to ppm_range if provided
        ax.set_xlim(ppm_range[1].high_ppm, ppm_range[1].low_ppm)
        ax.set_ylim(ppm_range[0].high_ppm, ppm_range[0].low_ppm)

    if traces is not None:
        traces = [Trace2D(spectra, *t) for t in traces]

    # check if we have to plot f1, f2 or f1+f2 external projections
    plot_projection_f1 = False
    plot_projection_f2 = False

    if projections is not None:
        projections = [Projection2D(spectra, *p) for p in projections]

        for projection in projections:
            if projection.is_external is True and projection.axis == "f1":
                plot_projection_f1 = True
            elif projection.is_external is True and projection.axis == "f2":
                plot_projection_f2 = True

    if plot_projection_f1 and plot_projection_f2:
        f1_proj_ax = ax.add_external_projection(axis="f1", scale_y=projection_f1_ratio)
        f2_proj_ax = ax.add_external_projection(axis="f2", scale_x=projection_f2_ratio)

        # correct f1_proj_ax to not overlap
        bbox = ax.get_position()
        width = bbox.x1 - bbox.x0

        bbox2 = f1_proj_ax.get_position()
        height2 = bbox2.y1 - bbox2.y0

        f1_proj_ax.set_position((bbox2.x0, bbox2.y0, width, height2))

    elif plot_projection_f1:
        f1_proj_ax = ax.add_external_projection(axis="f1", scale_y=projection_f1_ratio)
        f2_proj_ax = None
    elif plot_projection_f2:
        f1_proj_ax = None
        f2_proj_ax = ax.add_external_projection(axis="f2", scale_x=projection_f2_ratio)

    else:
        f1_proj_ax = None
        f2_proj_ax = None

    # styling of external projections, remove spines and axis
    for proj_ax in [f1_proj_ax, f2_proj_ax]:
        if proj_ax is not None:
            proj_ax.get_xaxis().set_visible(False)
            proj_ax.get_yaxis().set_visible(False)
            proj_ax.spines["top"].set_visible(False)  # Hide top spine
            proj_ax.spines["right"].set_visible(False)  # Hide right spine
            proj_ax.spines["left"].set_visible(False)  # Hide left spine
            proj_ax.spines["bottom"].set_visible(False)  # Hide bottom spine

    # pick peaks if requested
    run_pick_peaks = any(pick_peaks) if isinstance(pick_peaks, list) else bool(pick_peaks)
    if run_pick_peaks:
        for i, spectrum in enumerate(spectra):
            if isinstance(pick_peaks, list):
                do_pick_peaks = pick_peaks[i % len(pick_peaks)]
            else:
                do_pick_peaks = pick_peaks

            if isinstance(pick_peaks_kwargs, list):
                current_pick_peaks_kwargs = pick_peaks_kwargs[i % len(pick_peaks_kwargs)]
            else:
                current_pick_peaks_kwargs = pick_peaks_kwargs

            if do_pick_peaks:
                if "pick_sino" not in current_pick_peaks_kwargs.keys():
                    current_pick_peaks_kwargs["pick_sino"] = 4
                _pick_peak(spectrum, ppm_range, current_pick_peaks_kwargs)

    # mark peaks if requested
    run_mark_peaks = any(mark_peaks) if isinstance(mark_peaks, list) else bool(mark_peaks)
    if run_mark_peaks:
        for i, spectrum in enumerate(spectra):
            if isinstance(mark_peaks, list):
                do_mark_peaks = mark_peaks[i % len(mark_peaks)]
            else:
                do_mark_peaks = mark_peaks

            if isinstance(mark_peaks_kwargs, list):
                current_mark_peaks_kwargs = mark_peaks_kwargs[i % len(mark_peaks_kwargs)]
            else:
                current_mark_peaks_kwargs = mark_peaks_kwargs

            if do_mark_peaks:
                if "edgecolor" not in current_mark_peaks_kwargs.keys():
                    edgecolor = ax.collections[
                        len(ax.collections) - (i + 1)
                    ].get_edgecolor()[0]
                    current_mark_peaks_kwargs["edgecolor"] = edgecolor

                _mark_peaks(spectrum, ax, current_mark_peaks_kwargs)

    # set x and y tick_spacing
    if tick_spacing[0] is None:
        tick_spacing[0] = config.tick_spacing_2d[0]
    if tick_spacing[1] is None:
        tick_spacing[1] = config.tick_spacing_2d[1]

    ax.set_tick_spacing("f1", tick_spacing[0][0], tick_spacing[0][1])
    ax.set_tick_spacing("f2", tick_spacing[1][0], tick_spacing[1][1])

    # currently hard coded
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))

    # set labels after tick spacing has been set

    pos_f1 = f1_label_kwargs.pop("pos", None)
    pos_f2 = f2_label_kwargs.pop("pos", None)

    if pos_f1 is None:
        pos_f1 = (-0.1, 0.8)

    if show_f1_axis and show_f1_label:
        f1_label_text = f1_label_text.replace("[ppm]", """\n[ppm]""")
        ax.set2D_f1_label_text(
            f1_label_text,
            pos=pos_f1,
            f1_label_kwargs=f1_label_kwargs,
            f1_ticklabel_kwargs=f1_ticklabel_kwargs,
        )

    if show_f2_axis and show_f2_label:
        ax.set2D_f2_label_text(
            f2_label_text,
            pos=pos_f2,
            f2_label_kwargs=f2_label_kwargs,
            f2_ticklabel_kwargs=f2_ticklabel_kwargs,
        )
    #
    ax.xaxis.set_visible(show_f2_axis)
    ax.yaxis.set_visible(show_f1_axis)

    # check if one label or multiple labels
    if isinstance(label, str):
        label = [label]
    if isinstance(label_kwargs, dict):
        label_kwargs = [label_kwargs]

    # needs check for size
    if label is not None:
        _add_label(ax, label, label_kwargs)

    # add projections
    if projections is not None:
        _add_projections(spectra, projections, ax, f1_proj_ax, f2_proj_ax)

    # add traces
    if traces is not None:
        _add_traces(spectra, traces, ax)

    # testing

    if insets is not None:
        for inset in insets:
            if len(inset) == 3:
                excerpt, position, size = inset
                ax_kwargs = {}
            elif len(inset) == 4 and isinstance(inset[3], dict):
                excerpt, position, size, ax_kwargs = inset
            else:
                raise InvalidInsetError(inset)

            inset_ax = ax.add_inset(excerpt, position, size)

            plot2d(
                spectra,
                current_figure=figure,
                current_ax=inset_ax,
                ppm_range=excerpt,
                show_f1_axis=False,
                show_f2_axis=False,
                **{**kwargs, **ax_kwargs},
            )

    if markers_f1:
        ax_x_min, ax_x_max = ax.get_xlim()
        for i, marker in enumerate(markers_f1):
            if isinstance(markers_f1_kwargs, list):
                marker_kwargs = markers_f1_kwargs[i % len(markers_f1_kwargs)]
            elif isinstance(markers_f1_kwargs, dict):
                marker_kwargs = markers_f1_kwargs
            else:
                marker_kwargs = {}

            if "color" not in marker_kwargs.keys():
                marker_kwargs["color"] = (0.7, 0.7, 0.7, 0.7)
            if "linestyle" not in marker_kwargs.keys():
                marker_kwargs["linestyle"] = (0, (5, 1))

            ax.plot((ax_x_min, ax_x_max), (marker, marker), **marker_kwargs)

    if markers_f2:
        ax_y_min, ax_y_max = ax.get_ylim()
        for i, marker in enumerate(markers_f2):
            if isinstance(markers_f2_kwargs, list):
                marker_kwargs = markers_f2_kwargs[i % len(markers_f2_kwargs)]
            elif isinstance(markers_f2_kwargs, dict):
                marker_kwargs = markers_f2_kwargs
            else:
                marker_kwargs = {}

            if "color" not in marker_kwargs.keys():
                marker_kwargs["color"] = (0.7, 0.7, 0.7, 0.7)
            if "linestyle" not in marker_kwargs.keys():
                marker_kwargs["linestyle"] = (0, (5, 1))

            ax.plot((marker, marker), (ax_y_min, ax_y_max), **marker_kwargs)

    if legend is True:
        _add_legend(ax, legend_kwargs)

    figure.sca(ax)

    if set_axis_off:
        ax.set_axis_off()

    return figure, ax


def _add_projections(spectra, projections, ax, f1_proj_ax, f2_proj_ax):
    for projection in projections:
        if projection.is_external is True:
            trace_ppm = "projection_external"
        else:
            trace_ppm = "projection_internal"

        axis = projection.axis
        where = projection.spectrum_index
        projection_kwargs = projection.projection_kwargs

        if isinstance(projection.projection_kwargs, dict):
            projection_kwargs = [projection_kwargs] * len(spectra)

        if where == "all" or where == ["all"]:
            where = np.arange(len(spectra))

        plot_ax = None

        if trace_ppm == "projection_external":
            if axis == "f1":
                plot_ax = f1_proj_ax

                for d in projection_kwargs:
                    if "ppm_range" not in d.keys():
                        d["ppm_range"] = ax.get_xlim()

            if axis == "f2":
                plot_ax = f2_proj_ax
                for d in projection_kwargs:
                    if "ppm_range" not in d.keys():
                        d["ppm_range"] = ax.get_ylim()

        else:
            plot_ax = ax

        for i, spectrum in enumerate(spectra):
            if i in where:
                plot_ax.plot_1d_trace_or_projection_from_2d_spectrum(
                    spectrum, trace_ppm=trace_ppm, axis=axis, **projection_kwargs[i]
                )


def _add_traces(spectra, traces, ax):
    for trace in traces:
        trace_ppm = trace.chemical_shift.ppm
        axis = trace.axis
        where = trace.spectrum_index
        trace_kwargs = trace.trace_kwargs

        if isinstance(trace.trace_kwargs, dict):
            trace_kwargs = [trace_kwargs] * len(spectra)

        if where == "all" or where == ["all"]:
            where = np.arange(len(spectra))

        for i, spectrum in enumerate(spectra):
            if i in where:
                ax.plot_1d_trace_or_projection_from_2d_spectrum(
                    spectrum, trace_ppm=trace_ppm, axis=axis, **trace_kwargs[i]
                )


def _mark_peaks(spectrum, ax, mark_peaks_kwargs):

    if spectrum.peaklist is None or len(spectrum.peaklist) == 0:
        print("No peaks found in spectrum. Skipping marking of peaks.")
        return

    if "number_peaks" in mark_peaks_kwargs.keys():
        number_peaks = mark_peaks_kwargs["number_peaks"]
        del mark_peaks_kwargs["number_peaks"]
    else:
        number_peaks = False

    peaks = spectrum.peaklist

    x_val, y_val, marker_size_f1_ppm, marker_size_f2_ppm = zip(
        *[
            (
                peak.chemical_shifts[1].ppm,
                peak.chemical_shifts[0].ppm,
                peak.linewidths[0] / peak.frequencies[0],
                peak.linewidths[1] / peak.frequencies[1],
            )
            for peak in peaks
        ]
    )

    # Add rectangles manually at each (x, y) position
    peak_number = 0
    for x, y, w, h in zip(x_val, y_val, marker_size_f2_ppm, marker_size_f1_ppm):
        peak_number += 1

        f2_max, f2_min = ax.get_xlim()
        f1_max, f1_min = ax.get_ylim()

        scale2 = abs(f1_max - f1_min) / 100 * 1
        scale = abs(f2_max - f2_min) / 100 * 1

        rect = Rectangle(
            (x - (w + scale) / 2, y - (h + scale2) / 2),
            w + scale,
            h + scale2,
            facecolor="none",
            edgecolor=mark_peaks_kwargs["edgecolor"],
            linewidth=0.5,
        )
        ax.add_patch(rect)

        if number_peaks:
            ax.text(x, y, peak_number, fontsize=6)


def _pick_peak(spectrum, ppm_range, pick_peaks_kwargs):
    spectrum.pick_2d(ppm_range=ppm_range, sino=pick_peaks_kwargs["pick_sino"])


def _draw_peaks_from_list(
    peaks, ax, group
):  # for usage in GUI without a need for new picking
    if peaks is not None and len(peaks) > 0:
        if len(peaks[0]) < 8:
            x_val, y_val, marker_size_f1_ppm, marker_size_f2_ppm = zip(
                *[(a[1], a[0], 0.1, 0.1) for a in peaks]
            )
        else:
            x_val, y_val, marker_size_f1_ppm, marker_size_f2_ppm = zip(
                *[(a[2], a[1], a[7], a[8]) for a in peaks]
            )

        for x, y, w, h in zip(x_val, y_val, marker_size_f2_ppm, marker_size_f1_ppm):
            scale2 = 0.4  # here set to absolute values, otherwise boxes too small when called in zoomed mode
            scale = 0.05
            if group == "all":
                color = ax.collections[len(ax.collections) - 1].get_edgecolor()[0]
                # color = list(moxie_colors.complementary_color(color))
            elif group == "acc":
                color = ax.collections[len(ax.collections) - 1].get_edgecolor()[0]
                # color = list(moxie_colors.complementary_color(color))
            elif group == "scp":
                color = "red"
            elif group == "selected":
                color = "blue"
            rect = Rectangle(
                (x - (w + scale) / 2, y - (h + scale2) / 2),
                w + scale,
                h + scale2,
                facecolor="none",
                edgecolor=color,
                linewidth=0.5,
            )
            ax.add_patch(rect)
