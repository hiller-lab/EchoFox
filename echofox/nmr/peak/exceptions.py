
class PeakError(Exception):
    """Base class for peak-related exceptions."""


class InvalidPeakDimensionError(PeakError):
    """Raised when peak dimension is invalid."""

    def __init__(self, message):
        super().__init__(message)


class InvalidPeakIntensityError(PeakError):
    """Raised when peak intensity is invalid."""

    def __init__(self, value):
        message = f"Invalid peak intensity: {value!r}. Expected numeric value."
        super().__init__(message)
        self.value = value


class DimensionMismatchError(PeakError):
    """Raised when operations are performed on peaks with different dimensions."""

    def __init__(self, dim1, dim2):
        message = f"Dimension mismatch: cannot operate on {dim1}D and {dim2}D peaks"
        super().__init__(message)
        self.dim1 = dim1
        self.dim2 = dim2


class NucleusMismatchError(PeakError):
    """Raised when operations are performed on peaks with different nuclei."""

    def __init__(self, nuclei1, nuclei2, dimension=None):
        if dimension is not None:
            message = f"Nucleus mismatch at dimension {dimension}: {nuclei1[dimension]!r} != {nuclei2[dimension]!r}"
        else:
            message = f"Nucleus mismatch: {nuclei1} != {nuclei2}"
        super().__init__(message)
        self.nuclei1 = nuclei1
        self.nuclei2 = nuclei2
        self.dimension = dimension