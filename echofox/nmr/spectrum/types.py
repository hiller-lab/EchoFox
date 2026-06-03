from typing import Literal, TypeAlias

SpectrumFormat: TypeAlias = Literal["bruker", "pipe"]
SpectrumParam: TypeAlias = tuple[str, str, SpectrumFormat]
