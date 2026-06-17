"""
EchoFox - NMR Data Processing and Analysis Package
"""

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("echofox")
    except PackageNotFoundError:
        __version__ = "0.0.0"
except ImportError:
    __version__ = "0.0.0"

from .core import (
    Color,
    TimeRange,
    TimeValue,
)
from .echofoxplot import (
    SpectrumAxes,
    add_res_labels,
    bar_residues,
    bar_unassigned_residues,
    configure_font,
    draw_secondary_structure,
    fancy_legend,
    flatten_axs_list,
    get_irs,
    get_property_ratios,
    get_secondary_structure_map,
    get_vrs,
    highlight_res_labels,
    plot1d,
    plot2d,
    plot_cacb_deviations,
    plot_csps,
    plot_irs,
    savefig,
    subplots,
)
from .nmr import (
    ChemicalShift,
    NmrPeak,
    NmrSpectrum,
    PeakList,
    PeakListCollection,
    PpmRange,
    SyntheticPeak,
    get_cacb_deviations,
    read_spectra,
    smooth_cacb_deviations,
)
from .utils import (
    GreekLetters,
    convert_to_inches,
)

__all__ = [
    "__version__",
    # core
    "Color",
    "TimeRange",
    "TimeValue",
    # echofoxplot
    "add_res_labels",
    "bar_residues",
    "bar_unassigned_residues",
    "draw_secondary_structure",
    "get_secondary_structure_map",
    "plot_cacb_deviations",
    "get_irs",
    "get_property_ratios",
    "get_vrs",
    "highlight_res_labels",
    "plot1d",
    "plot2d",
    "plot_csps",
    "plot_irs",
    "subplots",
    "SpectrumAxes",
    "configure_font",
    "fancy_legend",
    "flatten_axs_list",
    "savefig",
    # nmr
    "ChemicalShift",
    "PpmRange",
    "NmrPeak",
    "PeakList",
    "PeakListCollection",
    "SyntheticPeak",
    "NmrSpectrum",
    "get_cacb_deviations",
    "smooth_cacb_deviations",
    "read_spectra",
    # utils
    "GreekLetters",
    "convert_to_inches",
]
