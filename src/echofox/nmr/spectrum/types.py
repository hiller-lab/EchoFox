from typing import Literal

SpectrumFormat: type = Literal["bruker", "pipe"]
SpectrumParam: type = tuple[str, str, SpectrumFormat]
