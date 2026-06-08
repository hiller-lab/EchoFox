import math
import re
from dataclasses import dataclass, field

from matplotlib import colors as mcolors

from .exceptions import (
    InvalidColorComponent,
    InvalidColorInput,
    InvalidHexError,
    UnsupportedColorFormat,
)


@dataclass
class Color:
    """
    Represents a Color object with support for various colors formats and conversions.

    The Color class allows for the creation of a color object using different formats,
    including hexadecimal strings, CSS colors strings, and individual RGB(A) components.
    It also supports HSL(A) and HSV(A) strings, plus explicit HSV constructors via
    ``Color.from_hsv(...)`` and ``Color.from_hsva(...)``.

    Color("#07F")
    Color("#0077FF80")
    Color("rgb(255, 0, 0)")
    Color("rgb(100%,0%,0%)")
    Color("rgba(0,128,255,0.5)")
    Color("rgba(0 128 255 / 50%)")
    Color("hsl(120, 50%, 50%)")
    Color("hsla(240, 100%, 50%, 0.25)")
    Color("hsv(120, 50%, 50%)")
    Color("hsva(240, 100%, 50%, 0.25)")
    Color("blue")
    Color("transparent")

    Color.from_hsl(120, 0.5, 0.5)
    Color.from_hsla(240, 1.0, 0.5, 0.25)
    Color.from_hsv(120, 0.5, 0.5)
    Color.from_hsva(240, 1.0, 0.5, 0.25)

    Color("#FF5733").adjust_hsv(val_mult=0.2)
    Color("#FF5733").adjust_hsl(lightness_mult=0.8)
    """

    _r: int = field(init=False, repr=False)
    _g: int = field(init=False, repr=False)
    _b: int = field(init=False, repr=False)
    _a: float = field(init=False, repr=False, default=1.0)

    HEX_LONG = re.compile(r"^#?([0-9A-Fa-f]{6})([0-9A-Fa-f]{2})?$")
    HEX_SHORT = re.compile(r"^#?([0-9A-Fa-f]{3})([0-9A-Fa-f]{1})?$")
    RGB_FUNC = re.compile(r"^rgba?\s*\((.+)\)$", re.IGNORECASE)
    HSL_FUNC = re.compile(r"^hsla?\s*\((.+)\)$", re.IGNORECASE)
    HSV_FUNC = re.compile(r"^hsva?\s*\((.+)\)$", re.IGNORECASE)

    NAMED = dict(mcolors.CSS4_COLORS)
    if "transparent" not in NAMED:
        NAMED["transparent"] = "#0000"

    def __init__(self, *args):
        """
        Accepts:
          - Color("#RGB" | "#RGBA" | "#RRGGBB" | "#RRGGBBAA")
          - Color("rgb(...)" | "rgba(...)" | "hsl(...)" | "hsla(...)" | "hsv(...)" | "hsva(...)")
          - Color(r, g, b)
          - Color(r, g, b, a)

        For numeric HSV input, use ``Color.from_hsv(h, s, v)`` or
        ``Color.from_hsva(h, s, v, a)`` to avoid ambiguity with RGB(A).
        """
        if len(args) == 1 and isinstance(args[0], str):
            self._init_from_string(args[0].strip())
        elif len(args) == 3:
            self._init_from_rgb(*args)
        elif len(args) == 4:
            self._init_from_rgba(*args)
        else:
            raise InvalidColorInput(args)

    # ------------ alternate constructors ------------
    @classmethod
    def from_hsl(cls, hue: float, saturation: float, lightness: float) -> "Color":
        obj = cls.__new__(cls)
        obj._init_from_hsl(hue, saturation, lightness)
        return obj

    @classmethod
    def from_hsla(cls, hue: float, saturation: float, lightness: float, a: float) -> "Color":
        obj = cls.__new__(cls)
        obj._init_from_hsla(hue, saturation, lightness, a)
        return obj

    @classmethod
    def from_hsv(cls, hue: float, saturation: float, value: float) -> "Color":
        obj = cls.__new__(cls)
        obj._init_from_hsv(hue, saturation, value)
        return obj

    @classmethod
    def from_hsva(cls, hue: float, saturation: float, value: float, a: float) -> "Color":
        obj = cls.__new__(cls)
        obj._init_from_hsva(hue, saturation, value, a)
        return obj

    # ------------ init helpers ------------
    def _init_from_rgb(self, r: int, g: int, b: int):
        self._check_rgb(r, g, b)
        self._r, self._g, self._b, self._a = r, g, b, 1.0

    def _init_from_rgba(self, r: int, g: int, b: int, a: float):
        self._check_rgb(r, g, b)
        self._check_a(a)
        self._r, self._g, self._b, self._a = int(r), int(g), int(b), float(a)

    def _init_from_hsl(self, hue: float, saturation: float, lightness: float):
        self._init_from_hsla(hue, saturation, lightness, 1.0)

    def _init_from_hsla(self, hue: float, saturation: float, lightness: float, a: float):
        self._check_unit_interval("saturation", saturation)
        self._check_unit_interval("lightness", lightness)
        self._check_a(a)
        r, g, b = self._hsl_to_rgb(float(hue) % 360.0, float(saturation), float(lightness))
        self._init_from_rgba(r, g, b, a)

    def _init_from_hsv(self, hue: float, saturation: float, value: float):
        self._init_from_hsva(hue, saturation, value, 1.0)

    def _init_from_hsva(self, hue: float, saturation: float, value: float, a: float):
        self._check_unit_interval("saturation", saturation)
        self._check_unit_interval("value", value)
        self._check_a(a)
        r, g, b = self._hsv_to_rgb(float(hue) % 360.0, float(saturation), float(value))
        self._init_from_rgba(r, g, b, a)

    def _init_from_hex(self, s: str):
        m = self.HEX_LONG.match(s)
        if m:
            rgb, a = m.groups()
            self._r = int(rgb[0:2], 16)
            self._g = int(rgb[2:4], 16)
            self._b = int(rgb[4:6], 16)
            self._a = int(a, 16) / 255 if a else 1.0
            return
        m = self.HEX_SHORT.match(s)
        if m:
            rgb, a = m.groups()
            self._r = int(rgb[0] * 2, 16)
            self._g = int(rgb[1] * 2, 16)
            self._b = int(rgb[2] * 2, 16)
            self._a = int(a * 2, 16) / 255 if a else 1.0
            return

        raise InvalidHexError(s)

    def _init_from_string(self, s: str):
        low = s.lower()
        if low in self.NAMED:
            return self._init_from_hex(self.NAMED[low])

        if s.startswith("#") or self.HEX_LONG.match(s) or self.HEX_SHORT.match(s):
            return self._init_from_hex(s)

        m = self.RGB_FUNC.match(s)
        if m:
            self._init_from_rgb_func(m.group(1))
            return

        m = self.HSL_FUNC.match(s)
        if m:
            self._init_from_hsl_func(m.group(1))
            return

        m = self.HSV_FUNC.match(s)
        if m:
            self._init_from_hsv_func(m.group(1))
            return

        raise UnsupportedColorFormat(s)

    # ------------ CSS arg splitting (now supports slash) ------------
    @staticmethod
    def _split_args(arg_str: str):
        """
        Split CSS function args supporting:
          - comma-separated: 'a, b, c[, d]'
          - space-separated: 'a b c[/ d]'
          - space+slash alpha: 'a b c / d'
        Returns list of 3 or 4 parts.
        """
        s = re.sub(r"\s+", " ", arg_str.strip())
        if "/" in s:  # space+slash alpha form
            left, right = [p.strip() for p in s.split("/", 1)]
            if "," in left:
                left_parts = [p.strip() for p in left.split(",") if p.strip()]
            else:
                left_parts = [p for p in left.split(" ") if p]
            return left_parts + [right]
        # no slash → either commas or spaces
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [p for p in s.split(" ") if p]

    def _init_from_rgb_func(self, inner: str):
        parts = self._split_args(inner)
        if len(parts) not in (3, 4):
            raise InvalidColorComponent("rgb/rgba", inner, "3 or 4 components")
        r = self._parse_rgb_component(parts[0])
        g = self._parse_rgb_component(parts[1])
        b = self._parse_rgb_component(parts[2])
        a = 1.0 if len(parts) == 3 else self._parse_alpha(parts[3])
        self._init_from_rgba(r, g, b, a)

    def _init_from_hsl_func(self, inner: str):
        parts = self._split_args(inner)
        if len(parts) not in (3, 4):
            raise InvalidColorComponent("hsl/hsla", inner, "3 or 4 components")
        hue = self._parse_hue(parts[0])
        saturation = self._parse_percent(parts[1])  # 0..1
        lightness = self._parse_percent(parts[2])  # 0..1
        a = 1.0 if len(parts) == 3 else self._parse_alpha(parts[3])
        self._init_from_hsla(hue, saturation, lightness, a)

    def _init_from_hsv_func(self, inner: str):
        parts = self._split_args(inner)
        if len(parts) not in (3, 4):
            raise InvalidColorComponent("hsv/hsva", inner, "3 or 4 components")
        hue = self._parse_hue(parts[0])
        saturation = self._parse_percent(parts[1])  # 0..1
        value = self._parse_percent(parts[2])  # 0..1
        a = 1.0 if len(parts) == 3 else self._parse_alpha(parts[3])
        r, g, b = self._hsv_to_rgb(hue, saturation, value)
        self._init_from_rgba(r, g, b, a)

    # ------------ parsers & converters ------------
    @staticmethod
    def _parse_rgb_component(tok: str) -> int:
        tok = tok.strip()
        if tok.endswith("%"):
            v = float(tok[:-1])
            if not (0.0 <= v <= 100.0):
                raise InvalidColorComponent("RGB percent", tok, "0%..100%")
            return round(v * 255 / 100.0)
        v = float(tok)
        if not (0.0 <= v <= 255.0):
            raise InvalidColorComponent("RGB value", tok, "0..255")
        return int(round(v))

    @staticmethod
    def _parse_percent(tok: str) -> float:
        tok = tok.strip()
        if not tok.endswith("%"):
            raise InvalidColorComponent("percentage", tok, "value ending with '%' like '50%'")
        v = float(tok[:-1]) / 100.0
        if not (0.0 <= v <= 1.0):
            raise InvalidColorComponent("percentage", tok, "0%..100%")
        return v

    @staticmethod
    def _parse_alpha(tok: str) -> float:
        tok = tok.strip()
        if tok.endswith("%"):
            v = float(tok[:-1]) / 100.0
        else:
            v = float(tok)
        if not (0.0 <= v <= 1.0):
            raise InvalidColorComponent("alpha", tok, "0..1 or 0%..100%")
        return v

    @staticmethod
    def _parse_hue(tok: str) -> float:
        tok = tok.strip().lower()
        if not tok:
            raise InvalidColorComponent("hue", tok, "number with optional deg/rad/turn")
        if tok.endswith("turn"):
            v = float(tok[:-4]) * 360.0
        elif tok.endswith("deg") or tok[-1].isdigit():
            v = float(tok[:-3]) if tok.endswith("deg") else float(tok)
        elif tok.endswith("rad"):
            v = float(tok[:-3]) * 180.0 / math.pi
        else:
            raise InvalidColorComponent("hue", tok, "number with optional deg/rad/turn")
        return v % 360.0

    @staticmethod
    def _hsl_to_rgb(hue_deg: float, saturation: float, lightness: float) -> tuple[int, int, int]:
        c = (1 - abs(2 * lightness - 1)) * saturation
        h = (hue_deg / 60.0) % 6
        x = c * (1 - abs(h % 2 - 1))
        if 0 <= h < 1:
            r1, g1, b1 = c, x, 0
        elif 1 <= h < 2:
            r1, g1, b1 = x, c, 0
        elif 2 <= h < 3:
            r1, g1, b1 = 0, c, x
        elif 3 <= h < 4:
            r1, g1, b1 = 0, x, c
        elif 4 <= h < 5:
            r1, g1, b1 = x, 0, c
        else:
            r1, g1, b1 = c, 0, x
        m = lightness - c / 2
        r = int(round((r1 + m) * 255))
        g = int(round((g1 + m) * 255))
        b = int(round((b1 + m) * 255))
        return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))

    @staticmethod
    def _hsv_to_rgb(hue_deg: float, saturation: float, value: float) -> tuple[int, int, int]:
        c = value * saturation
        h = (hue_deg / 60.0) % 6
        x = c * (1 - abs(h % 2 - 1))
        if 0 <= h < 1:
            r1, g1, b1 = c, x, 0
        elif 1 <= h < 2:
            r1, g1, b1 = x, c, 0
        elif 2 <= h < 3:
            r1, g1, b1 = 0, c, x
        elif 3 <= h < 4:
            r1, g1, b1 = 0, x, c
        elif 4 <= h < 5:
            r1, g1, b1 = x, 0, c
        else:
            r1, g1, b1 = c, 0, x
        m = value - c
        r = int(round((r1 + m) * 255))
        g = int(round((g1 + m) * 255))
        b = int(round((b1 + m) * 255))
        return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))

    @staticmethod
    def _rgb_to_hsl(r: int, g: int, b: int) -> tuple[float, float, float]:
        r_float = r / 255.0
        g_float = g / 255.0
        b_float = b / 255.0

        max_component = max(r_float, g_float, b_float)
        min_component = min(r_float, g_float, b_float)
        delta = max_component - min_component

        lightness = (max_component + min_component) / 2.0

        if delta == 0:
            hue = 0.0
            saturation = 0.0
        else:
            saturation = delta / (1.0 - abs(2.0 * lightness - 1.0))
            if max_component == r_float:
                hue = 60.0 * (((g_float - b_float) / delta) % 6)
            elif max_component == g_float:
                hue = 60.0 * (((b_float - r_float) / delta) + 2)
            else:
                hue = 60.0 * (((r_float - g_float) / delta) + 4)

        return hue % 360.0, saturation, lightness

    @staticmethod
    def _rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
        r_float = r / 255.0
        g_float = g / 255.0
        b_float = b / 255.0

        max_component = max(r_float, g_float, b_float)
        min_component = min(r_float, g_float, b_float)
        delta = max_component - min_component

        if delta == 0:
            hue = 0.0
        elif max_component == r_float:
            hue = 60.0 * (((g_float - b_float) / delta) % 6)
        elif max_component == g_float:
            hue = 60.0 * (((b_float - r_float) / delta) + 2)
        else:
            hue = 60.0 * (((r_float - g_float) / delta) + 4)

        saturation = 0.0 if max_component == 0 else delta / max_component
        value = max_component
        return hue % 360.0, saturation, value

    # ------------ validation ------------
    @staticmethod
    def _check_rgb(r, g, b):
        for n, v in zip("rgb", (r, g, b), strict=True):
            if not (isinstance(v, int) and 0 <= v <= 255):
                raise InvalidColorComponent(n, v, "int 0..255")

    @staticmethod
    def _check_a(a):
        if not (isinstance(a, (int, float)) and 0.0 <= float(a) <= 1.0):
            raise InvalidColorComponent("alpha", a, "0.0..1.0")

    @staticmethod
    def _check_unit_interval(component_name: str, value):
        if not (isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0):
            raise InvalidColorComponent(component_name, value, "0.0..1.0")

    @staticmethod
    def _clamp_unit(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    # ------------ adjustments ------------
    def adjust_hsv(
        self,
        hue_shift: float = 0.0,
        sat_mult: float = 1.0,
        val_mult: float = 1.0,
    ) -> "Color":
        """
        Return a new Color with adjusted HSV values.

        Parameters
        ----------
        hue_shift:
            Fraction of a full hue turn to add. ``0.1`` means +36 degrees,
            ``0.5`` means a 180-degree rotation.
        sat_mult:
            Saturation multiplier. Values are clamped to ``0.0..1.0``.
        val_mult:
            HSV value/brightness multiplier. Values are clamped to ``0.0..1.0``.

        Examples
        --------
        ``Color("#FF5733").adjust_hsv(val_mult=0.2).hex``
        """
        hue, saturation, value = self.hsv
        hue = (hue + float(hue_shift) * 360.0) % 360.0
        saturation = self._clamp_unit(saturation * sat_mult)
        value = self._clamp_unit(value * val_mult)
        return Color.from_hsva(hue, saturation, value, self._a)

    def adjust_hsl(
        self,
        hue_shift: float = 0.0,
        sat_mult: float = 1.0,
        lightness_mult: float = 1.0,
    ) -> "Color":
        """
        Return a new Color with adjusted HSL values.

        Parameters
        ----------
        hue_shift:
            Fraction of a full hue turn to add. ``0.1`` means +36 degrees,
            ``0.5`` means a 180-degree rotation.
        sat_mult:
            Saturation multiplier. Values are clamped to ``0.0..1.0``.
        lightness_mult:
            HSL lightness multiplier. Values are clamped to ``0.0..1.0``.

        Examples
        --------
        ``Color("#FF5733").adjust_hsl(lightness_mult=0.8).hex``
        """
        hue, saturation, lightness = self.hsl
        hue = (hue + float(hue_shift) * 360.0) % 360.0
        saturation = self._clamp_unit(saturation * sat_mult)
        lightness = self._clamp_unit(lightness * lightness_mult)
        return Color.from_hsla(hue, saturation, lightness, self._a)

    # ------------ properties ------------
    @property
    def rgb(self) -> tuple[int, int, int]:
        return (self._r, self._g, self._b)

    @property
    def rgba(self) -> tuple[int, int, int, float]:
        return (self._r, self._g, self._b, self._a)

    @property
    def hsl(self) -> tuple[float, float, float]:
        return self._rgb_to_hsl(self._r, self._g, self._b)

    @property
    def hsla(self) -> tuple[float, float, float, float]:
        h, s, l = self.hsl
        return h, s, l, self._a

    @property
    def hsv(self) -> tuple[float, float, float]:
        return self._rgb_to_hsv(self._r, self._g, self._b)

    @property
    def hsva(self) -> tuple[float, float, float, float]:
        h, s, v = self.hsv
        return h, s, v, self._a

    @property
    def hex(self) -> str:
        if self._a < 1.0:
            return f"#{self._r:02X}{self._g:02X}{self._b:02X}{round(self._a * 255):02X}"
        return f"#{self._r:02X}{self._g:02X}{self._b:02X}"

    def to_css_rgba(self) -> str:
        return f"rgba({self._r}, {self._g}, {self._b}, {self._a:.3f})"

    def to_css_hsla(self) -> str:
        h, s, l, a = self.hsla
        return f"hsla({h:.1f}, {s * 100:.1f}%, {l * 100:.1f}%, {a:.3f})"

    def to_css_hsva(self) -> str:
        h, s, v, a = self.hsva
        return f"hsva({h:.1f}, {s * 100:.1f}%, {v * 100:.1f}%, {a:.3f})"

    def __str__(self):
        return f"Color(rgb={self.rgb}, hsv={self.hsv}, a={self._a:.2f}, hex='{self.hex}')"
