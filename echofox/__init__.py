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
    configure_font,
    draw_secondary_structure,
    fancy_legend,
    flatten_axs_list,
    get_secondary_structure_map,
)
from .nmr import (
    ChemicalShift,
    NmrPeak,
    NmrSpectrum,
    PeakList,
    PeakListCollection,
    PpmRange,
    SyntheticPeak,
    read_spectra,
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
    "draw_secondary_structure",
    "get_secondary_structure_map",
    "SpectrumAxes",
    "configure_font",
    "fancy_legend",
    "flatten_axs_list",
    # nmr
    "ChemicalShift",
    "PpmRange",
    "NmrPeak",
    "PeakList",
    "PeakListCollection",
    "SyntheticPeak",
    "NmrSpectrum",
    "read_spectra",
    # utils
    "GreekLetters",
    "convert_to_inches",
]
