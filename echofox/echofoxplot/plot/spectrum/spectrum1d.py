# matplotlib
from matplotlib import rcParams
from matplotlib.figure import Figure

from echofox.echofoxplot.plot.spectrum.exceptions import LabelMismatchError

# matplotlib.use('TkAgg')
rcParams['pdf.fonttype'] = 42  # Configure Matplotlib to use TrueType fonts
rcParams['ps.fonttype'] = 42  # If exporting to PostScript as well

import numpy as np

# MiraPlot
from echofox.echofoxplot.axes.spectrum_axes import SpectrumAxes
from echofox.core.typing import Number
from echofox.nmr.chemical_shift  import ChemicalShift, PpmRange
from echofox.nmr.spectrum import NmrSpectrum
from echofox.nmr.spectrum  import read_spectra
from echofox.echofoxplot.config import config

from echofox.echofoxplot.plot.utils import _add_label
from echofox.echofoxplot.plot.utils import _add_legend


# typing


def plot1d(spec,
           ppm_range: tuple[Number, Number] | None = None,
           intensity_range: tuple[Number, Number] | None = None,
           tick_spacing: tuple[Number, Number] | None = config.tick_spacing_1d,
           f1_ticks: list[Number] | None = None,
           f1_ticklabels: list[str] | None = None,
           label: str | None = None,
           label_kwargs=None,

           show_f1_axis: bool = True,
           show_f1_label: bool = True,
           f1_label_kwargs: dict | None = None,
           f1_ticklabel_kwargs: dict | None = None,

           show_intensity_axis: bool = True,
           show_intensity_label: bool = True,
           intensity_label_kwargs: dict | None = None,
           intensity_ticklabel_kwargs: dict | None = None,

           current_figure: Figure | None = None,
           current_ax: SpectrumAxes | None = None,
           show_frame: bool = False,

           reverse_spectra: bool = False,

           normalize: bool = False,

           insets=None,  # think of how to do this
           legend: bool = False,
           legend_kwargs: dict | None = None,
           set_axis_off: bool = False,
           **kwargs):

    # set default lists and dicts
    if label_kwargs is None: label_kwargs = {}
    if f1_label_kwargs is None: f1_label_kwargs = {}
    if f1_ticklabel_kwargs is None: f1_ticklabel_kwargs = {}
    if intensity_label_kwargs is None: intensity_label_kwargs = {}
    if intensity_ticklabel_kwargs is None: intensity_ticklabel_kwargs = {}
    if legend_kwargs is None: legend_kwargs = {}

    # read spectra

    spectra = read_spectra(spec)
    if reverse_spectra:
        spectra.reverse()

    if not (len({x.ndim for x in spectra} & {1, 2}) == 1 and {x.ndim for x in spectra} <= {1, 2}) is True:
       raise Exception('Must be either 1D or 2D spectra.')


    # set current ax and figure
    figure = current_figure
    ax = current_ax

    # init variables
    f1_label_text = None


    # can we normalize all the spectra to max -1 to 1????
    # we can but is it nice or necessary?

    max_value = 0
    min_value = 0

    for i, spectrum in enumerate(spectra):

        if spectrum.data.ndim == 1:  # normal 1D spectrum
            max_val = np.max(spectrum.data)
            min_val = np.min(spectrum.data)
            max_value = max_val if max_val > max_value else max_value
            min_value = min_val if min_val < min_value else min_value

        elif spectrum.data.ndim == 2:  # treat as pseudo 2D spectrum

            for j in reversed(range(len(spectrum.data))):
                max_val = np.max(spectrum.data[j])
                min_val = np.min(spectrum.data[j])
                max_value = max_val if max_val > max_value else max_value
                min_value = min_val if min_val < min_value else min_value

    if normalize:
        factor = 1 / (abs(max_value) if abs(max_value) > abs(min_value) else abs(min_value))
    else:
        factor = 1



    # check offsets
    # create list with increasing values if no list is provided
    if 'intensity_offset' in kwargs.keys():
        if not isinstance(kwargs['intensity_offset'],list):
            if len(spectra) == 1:
                kwargs['intensity_offset'] = [kwargs['intensity_offset']]
            else:
                kwargs['intensity_offset'] = [kwargs['intensity_offset']*i for i in range(len(spectra))]


    if 'f1_offset' in kwargs.keys():
        if not isinstance(kwargs['f1_offset'],list):
            if len(spectra) == 1:
                kwargs['f1_offset'] = [kwargs['f1_offset']]
            else:
                kwargs['f1_offset'] = [kwargs['f1_offset'] * i for i in range(len(spectra))]


    # plot spectrum and extract variables
    for i, spectrum in enumerate(spectra):

        if spectrum.data.ndim == 1: # normal 1D spectrum

            new_dict = {}
            for key, value in kwargs.items():
                if isinstance(value, list):  # If the value is a list
                    if len(value) > 1:
                        new_dict[key] = value[i % len(value)]  # Get the element at index i (wrap around if needed)
                    else:
                        new_dict[key] = value[0]  # If only one element, take it
                else:
                    new_dict[key] = value  # If it's not a list, keep it as is

            ax.plot1d(spectrum,intensity_scale=factor, **new_dict)
            ax.set_xlim(spectrum.dimension_ranges[0].high_ppm, spectrum.dimension_ranges[0].low_ppm)

            if f1_label_text is None:
                f1_label_text = spectrum.get_label_text(0)
            elif not f1_label_text == spectrum.get_label_text(0):
                raise LabelMismatchError('f1', f1_label_text, spectrum.get_label_text(0))

        elif spectrum.data.ndim == 2: # treat as pseudo 2D spectrum

            for j in reversed(range(len(spectrum.data))):

                new_dict = {}
                for key, value in kwargs.items():
                    if isinstance(value, list):  # If the value is a list
                        if len(value) > 1:
                            new_dict[key] = value[i % len(value)]  # Get the element at index i (wrap around if needed)
                        else:
                            new_dict[key] = value[0]  # If only one element, take it
                    else:
                        new_dict[key] = value  # If it's not a list, keep it as is

                new_dict["f1_offset"] = new_dict["f1_offset"]*j
                new_dict["intensity_offset"] = new_dict["intensity_offset"]*j

                ax.plot1d(spectrum, intensity_scale=factor, index=j, **new_dict)

            ax.set_xlim(spectrum.dimension_ranges[1].high_ppm, spectrum.dimension_ranges[1].low_ppm)

            if f1_label_text is None:
                f1_label_text = spectrum.get_label_text(1)
            elif not f1_label_text == spectrum.get_label_text(1):
                raise LabelMismatchError('f1', f1_label_text, spectrum.get_label_text(1))



    if f1_ticks is not None and f1_ticklabels is None:
        ax.set_xticks(f1_ticks)
    elif f1_ticks is not None and f1_ticklabels is not None:
        if len(f1_ticks) != len(f1_ticklabels):
            raise Exception('Number of ticks and ticklabels must be equal.')
        ax.set_xticks(f1_ticks)
        ax.set_xticklabels(f1_ticklabels)
    else:
        # set x tick_spacing
        if tick_spacing is None:
            tick_spacing = config.tick_spacing_1d

        ax.set_tick_spacing('f1', tick_spacing[0], tick_spacing[1])

    # set spectrum limits to ppm_range if provided
    if ppm_range != (None):
        ppm_range = PpmRange(*ppm_range)
        # set spectrum limits to ppm_range if provided
        ax.set_xlim(ppm_range.high_ppm, ppm_range.low_ppm)

    if not intensity_range == None:
        ax.set_ylim(intensity_range[0], intensity_range[1])


    pos_f1 = f1_label_kwargs.pop('pos', None)
    if show_f1_axis and show_f1_label:
        ax.set1D_f1_label_text(f1_label_text, pos=pos_f1, f1_label_kwargs=f1_label_kwargs,
                               f1_ticklabel_kwargs=f1_ticklabel_kwargs)

    if show_intensity_axis and show_intensity_label:
        ax.set1D_intensity_label_text(label=label, pos=intensity_label_kwargs.pop('pos', None),
                                      intensity_label_kwargs=intensity_label_kwargs,
                                      intensity_ticklabel_kwargs=intensity_ticklabel_kwargs)

    ax.xaxis.set_visible(show_f1_axis)
    ax.yaxis.set_visible(show_intensity_axis)

    # check if one label or multiple labels
    if isinstance(label, str): label = [label]
    if isinstance(label_kwargs, dict): label_kwargs = [label_kwargs]

    # needs check for size
    if label is not None: _add_label(ax, label, label_kwargs)

    if legend is True: _add_legend(ax, legend_kwargs)

    ax.spines['top'].set_visible(show_frame)
    ax.spines['right'].set_visible(show_frame)
    ax.spines['left'].set_visible(show_frame)

    if set_axis_off: ax.set_axis_off()

    return figure, ax