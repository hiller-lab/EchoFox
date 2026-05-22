from __future__ import annotations

import numpy as np

from ..config import config
from .exceptions import NoiseEstimationError


def estimate_noise_level(self, method: str = None) -> float:
    if method is None:
        method = config.default_noise_method

    if method == 'std':
        return float(np.std(self._data))
    if method == 'median_absolute':
        return median_absolute_deviation(self)
    raise NoiseEstimationError(method, "Unknown method")


def median_absolute_deviation(self, k: float = 1.4826) -> float:
    d = np.ma.array(self._data).compressed()
    median = np.median(d)
    return float(k * np.median(np.abs(d - median)))
