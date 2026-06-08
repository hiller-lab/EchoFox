from __future__ import annotations


def to_dict(self) -> dict:
    result = {
        "name": self._name,
        "path": self._path,
        "spectrum_format": self._spectrum_format,
        "ndim": self._ndim,
        "shape": self._data.shape,
        "dimension_ranges": [r.to_tuple() for r in self._dimension_ranges],
        "nuclei": self._nuclei,
        "frequencies": self._frequencies,
        "is_pseudo_nd": self.is_pseudo_nd,
        "pseudo_axis_label": self._pseudo_axis_label,
        "pseudo_axis_unit": self._pseudo_axis_unit,
        "pseudo_axis_values": self._pseudo_axis_values,
        "source_files": self._source_files,
        "max_intensity": self.get_max_intensity(),
        "min_intensity": self.get_min_intensity(),
        "noise_level": self._noise_level,
        "processing_info": self._processing_info,
    }

    if self._acquisition_date:
        result["acquisition_date"] = self._acquisition_date.isoformat()

    result.update(self._metadata)

    return result


def spectrum_repr(self) -> str:
    nucleus_str = "×".join(self._nuclei) if self._nuclei else "unknown"
    shape_str = "×".join(str(s) for s in self._data.shape)
    name_str = f"'{self._name}', " if self._name else ""
    return f"NmrSpectrum({name_str}{self._ndim}D, {nucleus_str}, shape={shape_str})"


def spectrum_str(self) -> str:
    nucleus_str = "×".join(self._nuclei) if self._nuclei else "unknown nuclei"
    name_str = f"{self._name} - " if self._name else ""
    return f"{name_str}{self._ndim}D NMR Spectrum ({nucleus_str})"
