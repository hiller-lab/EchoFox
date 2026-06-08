from pathlib import Path
from typing import Literal

from echofox.core.typing import Number, is_typealias_instance

from .spectrum import NmrSpectrum

SpectrumFormat: type = Literal["bruker", "pipe"]
SpectrumParam: type = tuple[str | Number, str | Path, SpectrumFormat]
SpectrumParams: type = list[SpectrumParam]


def read_spectra(
    spec: SpectrumParam | SpectrumParams | NmrSpectrum | list[NmrSpectrum],
) -> list[NmrSpectrum]:
    # return spec if we are passed NmrSpectrum or a list[NmrSpectrum]

    if isinstance(spec, NmrSpectrum):
        return [spec]  # It's a single instance of a NmrSpectrum
    elif isinstance(spec, list) and all(isinstance(item, NmrSpectrum) for item in spec):
        return spec  # It's a list where all elements are NmrSpectrum

    # check if we are passed SpectrumParams or SpectrumParam
    if is_typealias_instance(spec, SpectrumParams) is False and is_typealias_instance(spec, SpectrumParam) is False:
        raise TypeError

    if is_typealias_instance(spec, SpectrumParam):
        spec = [spec]

    # create spectrum objects
    spectra = []
    for _i, spectrum_info in enumerate(spec):
        spectrum_name, spectrum_path, spectrum_format = spectrum_info

        spectrum = NmrSpectrum.from_file(spectrum_path, spectrum_format, spectrum_name)

        spectra.append(spectrum)

    return spectra
