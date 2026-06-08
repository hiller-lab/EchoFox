class ColorError(Exception):
    """Base class for colors-related exceptions."""


class InvalidHexError(ColorError):
    """Invalid hex string."""

    def __init__(self, value):
        super().__init__(f"Invalid HEX colors: {value} - use #RGB, #RGBA, #RRGGBB, or #RRGGBBAA")
        self.value = value


class UnsupportedColorFormat(ColorError):
    """Unsupported input format like 'cmyk()'."""

    def __init__(self, fmt):
        super().__init__(f"Unsupported colors format: {fmt}")
        self.format = fmt


class InvalidColorInput(ColorError):
    """Raised when Color() receives invalid or unsupported input."""

    def __init__(self, args_received):
        message = f"Color() expects a hex/CSS string, (r, g, b), or (r, g, b, a). Got: {args_received!r}"
        super().__init__(message)
        self.args_received = args_received


class InvalidColorComponent(ColorError):
    """Raised when a colors component (RGB, HSL, etc.) is outside the allowed range."""

    def __init__(self, component: str, value, expected: str):
        message = f"Invalid {component} value: {value!r}. Expected {expected}."
        super().__init__(message)
        self.component = component
        self.value = value
        self.expected = expected
