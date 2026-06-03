import re

# typing
from typing import Literal

import matplotlib.patches as patches
import matplotlib.projections as proj
import matplotlib.pyplot as plt

# numpy, scipy
import numpy as np

# matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import SubFigure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from echofox.core.colors import Color

# Import echofoxplot config
from echofox.echofoxplot.config import config
from echofox.nmr.chemical_shift import ChemicalShift
from echofox.nmr.spectrum import NmrSpectrum


class SpectrumAxes(Axes):
    """
    A custom subclass of Matplotlib's Axes to add custom functionality.
    """

    name = "SpectrumAxes"  # Required to register the custom Axes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.spectra = []

        self.cmap_positive = config.color2D
        self.cmap_negative = list(reversed(self.cmap_positive))
        self.color1D = config.color1D

    def clear(self):
        super().clear()
        self.spectra = []

    def plot1d(
        self,
        spectrum: NmrSpectrum,
        spectrum_color: str
        | tuple[float, float, float]
        | tuple[float, float, float, float]
        | None = None,
        f1_offset: float = 0,
        intensity_offset: float = 0,
        intensity_scale: float = 1,
        index: int = 0,
        **kwargs,
    ):

        spectrum_properties = {}

        if "linewidth" not in kwargs:
            if isinstance(self.figure, plt.Figure):
                fig_size = self.figure.get_size_inches()

            elif isinstance(self.figure, SubFigure):
                fig_size = self.figure.get_figure().get_size_inches()
            else:
                fig_size = None

            dpi = self.figure.get_dpi()
            if all(fig_size):
                fig_width_px = fig_size[0] * dpi
                fig_height_px = fig_size[1] * dpi

                # Calculate adaptive linewidth
                linewidth = max(0.1, min(fig_width_px, fig_height_px) / 8000)
                kwargs["linewidth"] = linewidth

            else:
                kwargs["linewidth"] = 0.5

        # spectrum colors
        if spectrum_color is None:
            spectrum_color = self.color1D[len(self.spectra) % len(self.color1D)]

        self.f1 = self.xaxis
        self.xaxis.set_inverted(True)
        self.yaxis.set_visible(False)

        # plot 1d
        spectrum_color = Color(spectrum_color).hex
        spectrum_properties["spectrum_color"] = spectrum_color
        kwargs["color"] = spectrum_color

        if spectrum.data.ndim == 1:
            self.plot(
                spectrum.get_ppm_axis(0) + f1_offset,
                (spectrum.data * intensity_scale) + intensity_offset,
                **kwargs,
            )
        elif spectrum.data.ndim == 2:  # treat as pseudo 2D
            self.plot(
                spectrum.get_ppm_axis(1) + f1_offset,
                (spectrum.data[index] * intensity_scale) + intensity_offset,
                **kwargs,
            )

        self.spectra.append([spectrum, spectrum_properties])

    def plot2d(
        self,
        spectrum: NmrSpectrum,
        *args,
        min_sino: int | float | None = None,
        contour_start_positive: int | float | None = None,
        contour_start_negative: int | float | None = None,
        contour_color_positive: str
        | tuple[float, float, float]
        | tuple[float, float, float, float]
        | None = None,
        contour_color_negative: str
        | tuple[float, float, float]
        | tuple[float, float, float, float]
        | None = None,
        contour_num: int = config.default_contour_levels,
        contour_factor: float = config.contour_level_factor,
        show: Literal["positive", "negative", "both"] = "both",
        **kwargs,
    ):

        spectrum_properties = {}
        spectrum_properties["name"] = spectrum.name

        self.f1 = self.yaxis
        self.f2 = self.xaxis

        # sino
        if min_sino is None:
            min_sino = spectrum.min_sino

        # calculate contour levels
        if contour_start_positive is None:
            contour_start = spectrum.noise_level * min_sino
        else:
            contour_start = contour_start_positive

        positive_contour_levels = contour_start * contour_factor ** np.arange(contour_num)
        positive_contour_levels.sort()

        if contour_start_negative is None:
            contour_start = -1 * spectrum.noise_level * min_sino
        else:
            contour_start = -1 * contour_start_positive

        negative_contour_levels = contour_start * contour_factor ** np.arange(contour_num)
        negative_contour_levels.sort()

        # contour colors
        if contour_color_positive is None:
            contour_color_positive = self.cmap_positive[
                len(self.spectra) % len(self.cmap_positive)
            ]

        if contour_color_negative is None:
            contour_color_negative = self.cmap_negative[
                len(self.spectra) % len(self.cmap_negative)
            ]

        contour_color_positive = Color(contour_color_positive).hex
        contour_color_negative = Color(contour_color_negative).hex

        # global parameters

        if "linewidths" not in kwargs:
            kwargs["linewidths"] = config.contour_linewidth

        # plot contours
        if show == "negative" or show == "both":
            kwargs["colors"] = contour_color_negative
            kwargs["extent"] = spectrum.extent
            self.contour(spectrum.data, negative_contour_levels, **kwargs)

            spectrum_properties["contour_color_negative"] = contour_color_negative

        if show == "positive" or show == "both":
            kwargs["colors"] = contour_color_positive
            kwargs["extent"] = spectrum.extent
            self.contour(spectrum.data, positive_contour_levels, **kwargs)

            spectrum_properties["contour_color_positive"] = contour_color_positive

        self.spectra.append([spectrum, spectrum_properties])

    def plot_1d_trace_or_projection_from_2d_spectrum(
        self,
        spectrum: NmrSpectrum,
        trace_ppm: str,
        *args,
        axis: Literal["f1", "f2"] = "f2",
        rel_scale: float | None = None,
        abs_scale: float | None = None,
        ppm_range: tuple[float, float] | str = "full",
        offset: float = 0,
        shift_to_edge: bool = False,
        **kwargs,
    ):

        if not isinstance(spectrum, NmrSpectrum):
            raise Exception("Spectrum must be of type Spectrum2D", 104)

        if axis not in ["f1", "f2"]:
            raise Exception("Axis must be either f1 or f2.", 103)

        # if scale is none --> calculate scale such that trace is 16% of the size of the whole spectrum
        if rel_scale is None and abs_scale is None:
            if axis == "f1":
                rel_scale = (
                    np.abs(spectrum.get_ppm_axis(0)[0] - spectrum.get_ppm_axis(0)[-1])
                    * 0.16
                )

            elif axis == "f2":
                rel_scale = (
                    np.abs(spectrum.get_ppm_axis(1)[0] - spectrum.get_ppm_axis(1)[-1])
                    * 0.16
                )

            if trace_ppm == "projection_external":
                rel_scale = 1

        # define ppm_range
        if isinstance(ppm_range, tuple):
            if not len(ppm_range) == 2:
                raise Exception(
                    "Input must be a tuple (min_ppm, max_ppm) or the string 'full'.", 105
                )

            if axis == "f1":
                end_index = np.argmin(
                    np.abs(spectrum.get_ppm_axis(1) - np.min(ppm_range))
                )
                start_index = np.argmin(
                    np.abs(spectrum.get_ppm_axis(1) - np.max(ppm_range))
                )

            elif axis == "f2":
                end_index = np.argmin(
                    np.abs(spectrum.get_ppm_axis(0) - np.min(ppm_range))
                )
                start_index = np.argmin(
                    np.abs(spectrum.get_ppm_axis(0) - np.max(ppm_range))
                )

        elif ppm_range == "full":
            if axis == "f1":
                start_index = 0
                end_index = len(spectrum.get_ppm_axis(1))

            elif axis == "f2":
                start_index = 0
                end_index = len(spectrum.get_ppm_axis(0))

        else:
            raise Exception(
                "Input must be a tuple (min_ppm, max_ppm) or the string 'full'.", 105
            )

        if trace_ppm == "projection_internal":
            shift_to_edge = True

        if isinstance(trace_ppm, float) or isinstance(trace_ppm, int):
            trace_ppm = str(
                str(trace_ppm) + " ppm"
            )  # ugly should deal with it differently

        if shift_to_edge:
            if trace_ppm == "projection_internal":
                if axis == "f1":
                    offset = (
                        max(self.get_ylim())
                        - (max(self.get_ylim()) - min(self.get_ylim())) * 0.01
                    )
                elif axis == "f2":
                    offset = (
                        min(self.get_xlim())
                        + (max(self.get_xlim()) - min(self.get_xlim())) * 0.01
                    )

            elif trace_ppm == "projection_external":
                pass

            elif isinstance(trace_ppm, str) and re.fullmatch(
                r"-?\d+(\.\d+)? ppm", trace_ppm
            ):
                if axis == "f1":
                    offset = (
                        float(trace_ppm.replace("ppm", "")) * -1
                        + max(self.get_ylim())
                        - (max(self.get_ylim()) - min(self.get_ylim())) * 0.01
                    )
                elif axis == "f2":
                    offset = (
                        float(trace_ppm.replace("ppm", "")) * -1
                        + min(self.get_xlim())
                        + (max(self.get_xlim()) - min(self.get_xlim())) * 0.01
                    )
            else:
                raise Exception(
                    "Trace must be either 'projection' or chemical shift.", 106
                )

        # get color that has been used for the 2d plot of that spectrum
        if "color" not in kwargs.keys():
            axes = self.figure.get_axes()
            for ax in axes:
                plotted_spectrum = [
                    plotted_spectrum
                    for plotted_spectrum in ax.spectra
                    if plotted_spectrum[0] == spectrum
                ]
                if plotted_spectrum:
                    for color in ["contour_color_positive", "contour_color_negative"]:
                        if color in plotted_spectrum[0][1].keys():
                            kwargs["color"] = plotted_spectrum[0][1].get(color)
                            break

        # default settings

        if "linewidth" not in kwargs.keys():
            kwargs["linewidth"] = 0.5

        # plot trace

        if axis == "f1":
            if trace_ppm == "projection_internal":
                if rel_scale:
                    trace = offset - (
                        spectrum.projection_f1
                        / np.max(spectrum.projection_f1)
                        * rel_scale
                    )
                else:
                    trace = offset - (spectrum.projection_f1 * abs_scale)

            elif trace_ppm == "projection_external":
                trace = spectrum.projection_f1

                d = np.ma.array(trace).compressed()
                median = np.median(d)
                x = 1.4826 * np.median(np.abs(d - median))
                trace = trace - x

                min_trace, max_trace = (
                    min(trace[start_index:end_index]),
                    max(trace[start_index:end_index]),
                )

                if abs(min_trace) > abs(max_trace) * 0.1:
                    min_trace = -1 * max_trace * 0.1

                if rel_scale:
                    trace = trace * rel_scale + offset
                else:
                    trace = trace * abs_scale + offset

            elif isinstance(trace_ppm, str) and re.fullmatch(
                r"-?\d+(\.\d+)? ppm", trace_ppm
            ):
                if rel_scale:
                    trace = (
                        float(trace_ppm.replace("ppm", ""))
                        + offset
                        - (
                            (spectrum.get_row(trace_ppm) / np.max(spectrum.data))
                            * rel_scale
                        )
                    )
                else:
                    trace = (
                        float(trace_ppm.replace("ppm", ""))
                        + offset
                        - ((spectrum.get_row(trace_ppm)) * abs_scale)
                    )

            else:
                raise Exception(
                    """Trace must be either 'projection' or chemical shift in [ppm] e.g '9.8 ppm' .""",
                    106,
                )

            self.plot(
                spectrum.get_ppm_axis(1)[start_index:end_index],
                trace[start_index:end_index],
                *args,
                **kwargs,
            )

        elif axis == "f2":
            if trace_ppm == "projection_internal":
                if rel_scale:
                    trace = (
                        spectrum.projection_f2
                        / np.max(spectrum.projection_f2)
                        * rel_scale
                    ) + offset
                else:
                    trace = (spectrum.projection_f2 * abs_scale) + offset

            elif trace_ppm == "projection_external":
                trace = spectrum.projection_f2

                d = np.ma.array(trace).compressed()
                median = np.median(d)
                x = 1.4826 * np.median(np.abs(d - median))
                trace = trace - x

                min_trace, max_trace = (
                    min(trace[start_index:end_index]),
                    max(trace[start_index:end_index]),
                )

                if abs(min_trace) > abs(max_trace) * 0.1:
                    min_trace = -1 * max_trace * 0.1

                if rel_scale:
                    trace = trace * rel_scale + offset
                else:
                    trace = trace * abs_scale + offset

            elif isinstance(trace_ppm, str) and re.fullmatch(
                r"-?\d+(\.\d+)? ppm", trace_ppm
            ):
                if rel_scale:
                    trace = (
                        (
                            (spectrum.get_column(trace_ppm) / np.max(spectrum.data))
                            * rel_scale
                        )
                        + float(trace_ppm.replace("ppm", ""))
                        + offset
                    )
                else:
                    trace = (
                        ((spectrum.get_column(trace_ppm)) * abs_scale)
                        + float(trace_ppm.replace("ppm", ""))
                        + offset
                    )
            else:
                raise Exception(
                    """Trace must be either 'projection' or chemical shift in [ppm] e.g '9.8 ppm' .""",
                    106,
                )

            self.plot(
                trace[start_index:end_index],
                spectrum.get_ppm_axis(0)[start_index:end_index],
                *args,
                **kwargs,
            )

        if trace_ppm == "projection_external":
            if axis == "f1":
                ylim = self.get_ylim()
                if ylim[1] >= max_trace:
                    max_trace = ylim[1]
                if ylim[0] <= min_trace:
                    min_trace = ylim[0]

                self.set_ylim(min_trace, max_trace)

            elif axis == "f2":
                xlim = self.get_xlim()
                if xlim[1] >= max_trace:
                    max_trace = xlim[1]
                self.set_xlim(min_trace, max_trace)

    def set1D_f1_label_text(
        self,
        label: str | None = None,
        pos: tuple[float, float] | None = None,
        f1_label_kwargs: dict = None,
        f1_ticklabel_kwargs: dict = None,
    ):

        if label is None:
            label = self.spectra[0][0].f1_label_text

        self.f1.set_label_text(label, **f1_label_kwargs)

        if pos is not None:
            self.f1.set_label_coords(*pos)

        xticks = self.get_xticks()
        xticks = [t for t in xticks if min(self.get_xlim()) <= t <= max(self.get_xlim())]
        self.set_xticks(xticks)

        ticklabels = [item.get_text() for item in self.get_xticklabels()]
        self.set_xticklabels(ticklabels, **f1_ticklabel_kwargs)

    def set1D_intensity_label_text(
        self,
        label: str | None = None,
        pos: tuple[float, float] | None = None,
        intensity_label_kwargs: dict = None,
        intensity_ticklabel_kwargs: dict = None,
    ):

        if label is None:
            label = "Intensity"

        self.yaxis.set_label_text(label, **intensity_label_kwargs)

        if pos is not None:
            self.yaxis.set_label_coords(*pos)

        self.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        yticks = self.get_yticks()
        yticks = [t for t in yticks if min(self.get_ylim()) <= t <= max(self.get_ylim())]
        self.set_yticks(yticks)

        # ticklabels = [item.get_text() for item in self.get_yticklabels()]
        # self.set_yticklabels(ticklabels, **intensity_ticklabel_kwargs) #doesnt work yet

    def set2D_f1_label_text(
        self,
        label: str | None = None,
        pos: tuple[float, float] | None = None,
        f1_label_kwargs: dict = None,
        f1_ticklabel_kwargs: dict = None,
        debug=False,
    ):

        if f1_label_kwargs is None:
            f1_label_kwargs = {}

        f1_label_kwargs["rotation"] = 0
        f1_label_kwargs["multialignment"] = "center"
        f1_label_kwargs["bbox"] = dict(edgecolor="white", facecolor="white", alpha=0)

        if label is None:
            label = self.spectra[0][0].f1_label_text

        self.f1.set_label_text(label, **f1_label_kwargs)

        if pos is None:
            pos = (-0.09, 0.8)

        if pos is not None:
            self.f1.set_label_coords(*pos)

        # check for overlap with tick labels

        # we have to set the current yticks first to update bbox correctly
        yticks = self.get_yticks()
        yticks = [t for t in yticks if min(self.get_ylim()) <= t <= max(self.get_ylim())]
        self.set_yticks(yticks)

        ticks = [t for t in self.get_yticklabels()]
        ticklabels = [item.get_text() for item in self.get_yticklabels()]

        label_bbox = self.f1.label.get_window_extent(
            self.figure.canvas.get_renderer()
        ).transformed(self.figure.transFigure.inverted())

        # Check against tick labels
        for i, tick in enumerate(ticks):
            if tick.get_text():
                tick_bbox = tick.get_window_extent(
                    self.figure.canvas.get_renderer()
                ).transformed(self.get_figure().transFigure.inverted())
                if self._bbox_overlap(tick_bbox, label_bbox):
                    ticklabels[i] = ""

        box = (
            self.spines["left"]
            .get_window_extent(self.figure.canvas.get_renderer())
            .transformed(self.get_figure().transFigure.inverted())
        )

        if debug:
            print(self._bbox_overlap(box, label_bbox, debug=True))

        self.set_yticklabels(ticklabels, **f1_ticklabel_kwargs)

    def set2D_f2_label_text(
        self,
        label: str | None = None,
        pos: tuple[float, float] | None = None,
        f2_label_kwargs: dict = None,
        f2_ticklabel_kwargs: dict = None,
    ) -> None:

        if f2_label_kwargs is None:
            f2_label_kwargs = {}

        if label is None:
            label = self.spectra[0][0].f2_label_text

        self.f2.set_label_text(label, **f2_label_kwargs)

        if pos is not None:
            self.f2.set_label_coords(*pos)

        xticks = self.get_xticks()
        xticks = [t for t in xticks if min(self.get_xlim()) <= t <= max(self.get_xlim())]
        self.set_xticks(xticks)

        ticklabels = [item.get_text() for item in self.get_xticklabels()]
        self.set_xticklabels(ticklabels, **f2_ticklabel_kwargs)

    def _bbox_overlap(self, rect1, rect2, debug=False):
        if debug:
            self._plot_bbox_figure(self.figure, rect1)
            self._plot_bbox_figure(self.figure, rect2)

        return rect1.overlaps(rect2)

    def _plot_bbox_figure(self, fig, bbox, color="r", label="BBox"):
        """Plot a bounding box in figure coordinates."""
        x0, y0, width, height = bbox.bounds
        rect = patches.Rectangle(
            (x0, y0),
            width,
            height,
            transform=fig.transFigure,
            fill=False,
            edgecolor=color,
            linewidth=2,
            label=label,
        )
        fig.patches.append(rect)

    def _get_figure_relative_coordinates(self, x, y):
        return (
            self.get_figure()
            .transFigure.inverted()
            .transform(self.transData.transform((x, y)))
        )

    def set_tick_spacing(
        self, axis: Literal["f1", "f2"] = "f1", major: float = 5, minor: float = 1
    ):

        if minor >= major:
            raise Exception("Minor spacing must be smaller than major spacing.", 101)

        if axis not in ["f1", "f2"]:
            raise Exception("Axis must be either f1 or f2.", 102)

        if axis == "f1":
            self.f1.set_major_locator(plt.MultipleLocator(major))
            self.f1.set_minor_locator(plt.MultipleLocator(minor))
        elif axis == "f2":
            self.f2.set_major_locator(plt.MultipleLocator(major))
            self.f2.set_minor_locator(plt.MultipleLocator(minor))

    def add_external_projection(self, axis="f1", scale_x=0.85, scale_y=0.85, **kwargs):

        if scale_x < 0.1 or scale_x > 0.9:
            scale_x = 0.85  # avoid stupid settings

        if scale_y < 0.1 or scale_y > 0.9:
            scale_y = 0.85  # avoid stupid settings

        scale_factor_x = scale_x
        scale_factor_y = scale_y

        inset_ax = None

        # Get the bounding box of the axis in figure coordinates
        bbox = self.get_position()

        if axis == "f1":
            # Extract position and size
            x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
            width = x1 - x0
            height = y1 - y0

            new_width = width * 1
            new_height = height * scale_factor_y

            self.set_position((x0, y0, new_width, new_height))

            min_y, max_y = self.get_ylim()
            sw_y = abs(min_y - max_y)

            position_x0 = self.get_xlim()[0]
            position_y0 = self.get_ylim()[1]

            position_x1 = self.get_xlim()[1]
            position_y1 = self.get_ylim()[1] - sw_y * (1 - scale_factor_y)

            inset_x0, inset_y0 = self._get_figure_relative_coordinates(
                position_x0, position_y0
            )
            inset_x1, inset_y1 = self._get_figure_relative_coordinates(
                position_x1, position_y1
            )

            inset_ax = self.get_figure().add_axes(
                [inset_x0, inset_y0, inset_x1 - inset_x0, inset_y1 - inset_y0],
                axes_class=SpectrumAxes,
            )

            inset_ax.set_xlim(position_x0, position_x1)

        elif axis == "f2":
            # Extract position and size
            x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
            width = x1 - x0
            height = y1 - y0

            new_width = width * scale_factor_x
            new_height = height * 1

            self.set_position((x0, y0, new_width, new_height))

            min_x, max_x = self.get_xlim()
            sw_x = abs(min_x - max_x)

            position_x0 = self.get_xlim()[1]
            position_y0 = self.get_ylim()[0]

            position_x1 = self.get_xlim()[1] - sw_x * (1 - scale_factor_x)
            position_y1 = self.get_ylim()[1]

            inset_x0, inset_y0 = self._get_figure_relative_coordinates(
                position_x0, position_y0
            )
            inset_x1, inset_y1 = self._get_figure_relative_coordinates(
                position_x1, position_y1
            )

            inset_ax = self.get_figure().add_axes(
                [inset_x0, inset_y0, inset_x1 - inset_x0, inset_y1 - inset_y0],
                axes_class=SpectrumAxes,
            )

            inset_ax.set_ylim(position_y0, position_y1)

        return inset_ax

    def add_inset(self, excerpt, position, size, **kwargs):

        position_x0, position_y0 = position
        position_x1, position_y1 = position[0] - size[0], position[1] - size[1]

        inset_x0, inset_y0 = self._get_figure_relative_coordinates(
            position_x0, position_y0
        )
        inset_x1, inset_y1 = self._get_figure_relative_coordinates(
            position_x1, position_y1
        )

        inset_ax = self.get_figure().add_axes(
            [inset_x0, inset_y0, inset_x1 - inset_x0, inset_y1 - inset_y0],
            axes_class=SpectrumAxes,
        )

        (x1, y1), (x2, y2) = zip(excerpt[1], excerpt[0], strict=True)
        x1 = ChemicalShift(x1).ppm
        y1 = ChemicalShift(y1).ppm
        x2 = ChemicalShift(x2).ppm
        y2 = ChemicalShift(y2).ppm

        if x1 < x2:
            x1, x2 = x2, x1
        if y1 < y2:
            y1, y2 = y2, y1

        inset_ax.set_xlim(x1, x2)
        inset_ax.set_ylim(y1, y2)

        # Create a rectangle
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            edgecolor="black",
            facecolor=(1, 1, 1, 0.5),
            linewidth=0.5,
        )

        left = False
        right = False
        top = False
        bottom = False

        if max(position_x0, position_x1) < x1 and max(position_x0, position_x1) < x2:
            left = False
            right = True

            p4 = (x1, y1)
            p3 = (x2, y1)
            p2 = (x1, y2)
            p1 = (x2, y2)

        elif min(position_x0, position_x1) > x1 and min(position_x0, position_x1) > x2:
            left = True
            right = False

            p1 = (x1, y1)
            p2 = (x2, y1)
            p3 = (x1, y2)
            p4 = (x2, y2)

        elif max(position_y0, position_y1) < y1 and max(position_y0, position_y1) < y2:
            top = True
            bottom = False

            p1 = (x1, y2)
            p2 = (x1, y1)
            p3 = (x2, y2)
            p4 = (x2, y1)

        elif min(position_y0, position_y1) > y1 and min(position_y0, position_y1) > y2:
            top = False
            bottom = True

            p1 = (x1, y1)
            p2 = (x1, y2)
            p3 = (x2, y1)
            p4 = (x2, y2)

        if left is False and right is False and top is False and bottom is False:
            raise Exception("Inset position is overlapping with its own region.", 107)

        fig_relative_coords = (
            self.get_figure()
            .transFigure.inverted()
            .transform(inset_ax.transData.transform(p2))
        )
        display_coords = inset_ax.get_figure().transFigure.transform(fig_relative_coords)
        data_coords = self.transData.inverted().transform(display_coords)

        (xx2, yy2) = data_coords

        # Create a Line2D object
        line = Line2D(
            [p1[0], xx2], [p1[1], yy2], color="black", linewidth=0.5, linestyle="-"
        )

        # Add the line to the axes
        self.add_line(line)

        fig_relative_coords = (
            inset_ax.get_figure()
            .transFigure.inverted()
            .transform(inset_ax.transData.transform(p4))
        )
        display_coords = self.get_figure().transFigure.transform(fig_relative_coords)
        data_coords = self.transData.inverted().transform(display_coords)

        (xx2, yy2) = data_coords

        # Create a Line2D object
        line = Line2D(
            [p3[0], xx2], [p3[1], yy2], color="black", linewidth=0.5, linestyle="-"
        )

        # Add the line to the axes
        self.add_line(line)

        # Add the rectangle to the axes
        self.add_patch(rect)

        return inset_ax


proj.register_projection(SpectrumAxes)
