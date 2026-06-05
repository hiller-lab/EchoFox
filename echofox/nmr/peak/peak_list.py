import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .peak import NmrPeak


class PeakList:
    """
    Container for managing collections of NMR peaks.

    PeakList provides a list-like interface for managing multiple Peak objects,
    with additional methods for searching, filtering, sorting, and analyzing
    peak collections.

    Examples:
        # Create empty peak list
        peak_list = PeakList()

        # Add peaks
        peak_list.add(Peak([7.26], nuclei=['1H']))
        peak_list.add(Peak([8.5, 120.2], nuclei=['1H', '15N']))

        # Access peaks
        first_peak = peak_list[0]
        num_peaks = len(peak_list)

        # Iterate over peaks
        for peak in peak_list:
            print(peak)

        # Filter peaks
        h_peaks = peak_list.filter_by_nucleus('1H', dimension=0)
        assigned_peaks = peak_list.filter_assigned()

        # Find closest peak to a position
        closest = peak_list.find_closest([7.25, 120.0])

        # Sort by chemical shift
        peak_list.sort_by_shift(dimension=0)

    Args:
        peaks: Optional initial list of peaks
        name: Optional name for the peak list
    """

    def __init__(
        self,
        peaks: list[NmrPeak] | None = None,
        name: str | None = None,
        metadata: dict | None = None,
    ):
        self._peaks: list[NmrPeak] = []
        self.name = name
        self._metadata = metadata or {}
        self._experiment_id: int | None = None

        if peaks is not None:
            for peak in peaks:
                self.add(peak)

    @property
    def peaks(self) -> list[NmrPeak]:
        """Returns a copy of the peak list."""
        return self._peaks.copy()

    @property
    def metadata(self) -> dict:
        """Return a copy of peak-list metadata."""
        return self._metadata.copy()

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        *,
        name: str | None = None,
        format: str | None = None,
        **kwargs,
    ) -> "PeakList":
        """
        Read a peak list from Excel, NEF, or JSON.
        """
        file_path = Path(file_path)

        if format is None:
            format = file_path.suffix.lower().lstrip(".")

        if format in {"xlsx", "xls"}:
            return cls.from_excel(file_path, name=name, **kwargs)

        if format == "nef":
            return cls.from_nef(file_path, name=name, **kwargs)

        if format == "json":
            return cls.load(file_path)

        raise ValueError(f"Unsupported peak-list format: {format!r}")

    @classmethod
    def from_excel(
        cls,
        file_path: str | Path,
        *,
        name: str | None = None,
        sheet_name: str | int = 0,
        chemical_shift_columns: Sequence[str] | None = None,
        nuclei: Sequence[str] | None = None,
        nucleus_columns: Sequence[str] | None = None,
        assignment_columns: Sequence[str] | None = None,
        frequency_columns: Sequence[str] | None = None,
        linewidth_columns: Sequence[str] | None = None,
        gaussian_sigma_columns: Sequence[str] | None = None,
        lorentzian_gamma_columns: Sequence[str] | None = None,
        intensity_column: str | None = None,
        volume_column: str | None = None,
        metadata: dict | None = None,
        **read_excel_kwargs,
    ) -> "PeakList":
        """
        Create a PeakList from an Excel table.

        Example:
            peaks = PeakList.from_excel(
                "peaks.xlsx",
                chemical_shift_columns=["HN", "N"],
                nuclei=["1H", "15N"],
                assignment_columns=["H_assignment", "N_assignment"],
                intensity_column="height",
            )
        """
        file_path = Path(file_path)
        df = pd.read_excel(file_path, sheet_name=sheet_name, **read_excel_kwargs)

        return cls.from_dataframe(
            df,
            name=name or file_path.stem,
            chemical_shift_columns=chemical_shift_columns,
            nuclei=nuclei,
            nucleus_columns=nucleus_columns,
            assignment_columns=assignment_columns,
            frequency_columns=frequency_columns,
            linewidth_columns=linewidth_columns,
            gaussian_sigma_columns=gaussian_sigma_columns,
            lorentzian_gamma_columns=lorentzian_gamma_columns,
            intensity_column=intensity_column,
            volume_column=volume_column,
            metadata={
                "source": str(file_path),
                "format": "excel",
                "sheet_name": sheet_name,
                **(metadata or {}),
            },
        )

    @staticmethod
    def _clean_value(value: Any) -> Any:
        """Normalize pandas/CCPNMR missing values."""
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass

        if isinstance(value, str):
            value = value.strip()
            if value in {"", ".", "?", "nan", "NaN", "None", "none"}:
                return None

        return value

    @classmethod
    def _as_float(cls, value: Any) -> float | None:
        value = cls._clean_value(value)
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _safe_int(value) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _collapse_if_identical(values: list):
        """
        Return the single value if all values in the list are identical.
        Otherwise return the original list.

        Examples:
            ["A", "A"] -> "A"
            ["H", "N"] -> ["H", "N"]
        """
        if values and all(value == values[0] for value in values):
            return values[0]

        return values

    @classmethod
    def from_ccpnmr_table(
        cls,
        file_path: str | Path,
        name: str | None = None,
        is_raw_ccpnmr_table: bool | None = None,
        sheet_name: str | int = 0,
        metadata: dict | None = None,
        **read_excel_kwargs,
    ) -> "PeakList":
        """
        Read an Excel peak table exported from CCPNMR and convert it to a PeakList.

        Supported CCPNMR table styles:

        Standard/exported table:
            Assign F1, Assign F2, Pos F1, Pos F2, LW F1 (Hz), LW F2 (Hz),
            Height, S/N, Volume, ClusterId, Merit, Annotation, Comment

        Raw table:
            Assign_F1, Assign_F2, comment, height, ppmLineWidth_F1,
            ppmLineWidth_F2, ppmPosition_F1, ppmPosition_F2, volume

        Assignment strings are expected to look like:
            WT_FL_A.2.GLY.H

        Raw tables may contain a leading "NA:", e.g.:
            NA:WT_FL_A.100.ALA.H

        The assignment is parsed into per-dimension extra NmrPeak properties:
            chain
            residue_index
            residue_type
            atom

        For example:
            peak.chain == ["WT_FL_A", "WT_FL_A"]
            peak.residue_index == [2, 2]
            peak.residue_type == ["GLY", "GLY"]
            peak.atom == ["H", "N"]
        """
        file_path = Path(file_path)
        df = pd.read_excel(file_path, sheet_name=sheet_name, **read_excel_kwargs)

        if is_raw_ccpnmr_table is None:
            is_raw_ccpnmr_table = any(str(col).startswith("Assign_F") for col in df.columns)

        if is_raw_ccpnmr_table:
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            # Drop raw table specific columns
            for col_pattern in [
                "aliasing_F",
                "axisCode_F",
                "className",
                "id",
                "isDeleted",
                "longPid",
                "pid",
                "pointPosition_F",
                "position_F",
                "serial",
                "shortClassName",
                "tartPosition_F1",
                "tartPosition_F2",
            ]:
                df = df[df.columns.drop(list(df.filter(regex=col_pattern)))]
        else:
            df = df.drop(columns="#")

        df = df.sort_values(df.columns[0])

        dimensions = cls._infer_ccpnmr_dimensions(df.columns, is_raw_ccpnmr_table=is_raw_ccpnmr_table)

        if not dimensions:
            raise ValueError(
                "Could not infer CCPNMR dimensions. Expected columns like "
                "'Assign F1' / 'Pos F1' or 'Assign_F1' / 'ppmPosition_F1'."
            )

        peaks: list[NmrPeak] = []

        for _, row in df.iterrows():
            chemical_shifts: list[float] = []
            assignments: list[str | None] = []
            nuclei: list[str | None] = []
            chains: list[str | None] = []
            residue_indices: list[int | None] = []
            residue_types: list[str | None] = []
            atoms: list[str | None] = []
            linewidths_hz: list[float] = []
            ppm_linewidths: list[float] = []

            skip_row = False

            for dim in dimensions:
                assignment_col = cls._ccpnmr_assignment_column(dim, is_raw_ccpnmr_table)
                position_col = cls._ccpnmr_position_column(dim, is_raw_ccpnmr_table)
                linewidth_col = cls._ccpnmr_linewidth_column(dim, is_raw_ccpnmr_table)

                position = cls._as_float(row[position_col])

                if position is None:
                    skip_row = True
                    break

                assignment_raw = cls._clean_value(row[assignment_col])
                parsed_assignment = cls._parse_ccpnmr_assignment(assignment_raw)

                chemical_shifts.append(position)
                assignments.append(parsed_assignment["assignment"])
                nuclei.append(cls._nucleus_from_atom_name(parsed_assignment["atom"]))

                chains.append(parsed_assignment["chain"])
                residue_indices.append(parsed_assignment["residue_index"])
                residue_types.append(parsed_assignment["residue_type"])
                atoms.append(parsed_assignment["atom"])

                if linewidth_col in row.index:
                    linewidth_value = cls._as_float(row[linewidth_col])

                    if linewidth_value is not None:
                        if is_raw_ccpnmr_table:
                            ppm_linewidths.append(linewidth_value)
                        else:
                            linewidths_hz.append(linewidth_value)

            if skip_row:
                continue

            # Do not pass incomplete nuclei lists to NmrPeak, because NmrPeak
            # converts nucleus labels to strings.
            peak_nuclei = nuclei if all(nucleus is not None for nucleus in nuclei) else None

            intensity_col = cls._ccpnmr_first_existing_column(
                df.columns,
                ["Height", "height"],
            )
            volume_col = cls._ccpnmr_first_existing_column(
                df.columns,
                ["Volume", "volume"],
            )

            intensity = cls._as_float(row[intensity_col]) if intensity_col else None
            volume = cls._as_float(row[volume_col]) if volume_col else None

            extra_properties = {
                "chain": cls._collapse_if_identical(chains),
                "residue_index": cls._collapse_if_identical(residue_indices),
                "residue_type": cls._collapse_if_identical(residue_types),
                "atom": atoms,
            }

            if ppm_linewidths:
                # Raw CCPNMR table column says ppmLineWidth_Fn, so keep this
                # separate from NmrPeak.linewidths, which are documented as Hz.
                extra_properties["ppm_linewidths"] = ppm_linewidths

            # Keep useful non-dimensional CCPNMR columns as metadata,
            # but do not duplicate values already mapped to NmrPeak properties.
            used_columns = {col for col in (intensity_col, volume_col) if col is not None}
            ccpnmr_metadata = {}
            for col in df.columns:
                col_str = str(col)

                if col in used_columns:
                    continue

                if cls._is_ccpnmr_dimension_column(col_str):
                    continue

                value = cls._clean_value(row[col])

                if value is not None:
                    ccpnmr_metadata[col_str] = value

            extra_properties.update(ccpnmr_metadata)

            peak = NmrPeak(
                chemical_shifts=chemical_shifts,
                nuclei=peak_nuclei,
                intensity=intensity,
                volume=volume,
                linewidths=linewidths_hz if linewidths_hz else None,
                assignments=assignments if any(a is not None for a in assignments) else None,
                **extra_properties,
            )

            peaks.append(peak)

        return cls(
            peaks,
            name=name or file_path.stem,
            metadata={
                "source": str(file_path),
                "format": "ccpnmr_table",
                "is_raw_ccpnmr_table": is_raw_ccpnmr_table,
                "sheet_name": sheet_name,
                **(metadata or {}),
            },
        )

    @staticmethod
    def _ccpnmr_assignment_column(dim: int, is_raw_ccpnmr_table: bool) -> str:
        if is_raw_ccpnmr_table:
            return f"Assign_F{dim}"

        return f"Assign F{dim}"

    @staticmethod
    def _ccpnmr_position_column(dim: int, is_raw_ccpnmr_table: bool) -> str:
        if is_raw_ccpnmr_table:
            return f"ppmPosition_F{dim}"

        return f"Pos F{dim}"

    @staticmethod
    def _ccpnmr_linewidth_column(dim: int, is_raw_ccpnmr_table: bool) -> str:
        if is_raw_ccpnmr_table:
            return f"ppmLineWidth_F{dim}"

        return f"LW F{dim} (Hz)"

    @classmethod
    def _infer_ccpnmr_dimensions(
        cls,
        columns,
        *,
        is_raw_ccpnmr_table: bool,
    ) -> list[int]:
        """
        Infer dimensions from CCPNMR assignment and position columns.

        Returns:
            Sorted dimension numbers, e.g. [1, 2] or [1, 2, 3, 4].
        """
        columns = set(str(col) for col in columns)
        dimensions: list[int] = []

        if is_raw_ccpnmr_table:
            pattern = re.compile(r"^Assign_F(\d+)$")
        else:
            pattern = re.compile(r"^Assign F(\d+)$")

        for col in columns:
            match = pattern.match(col)

            if not match:
                continue

            dim = int(match.group(1))
            position_col = cls._ccpnmr_position_column(dim, is_raw_ccpnmr_table)

            if position_col in columns:
                dimensions.append(dim)

        return sorted(dimensions)

    @staticmethod
    def _parse_ccpnmr_assignment(value) -> dict:
        """
        Parse CCPNMR assignment strings.

        Examples:
            WT_FL_A.2.GLY.H
            NA:WT_FL_A.100.ALA.H

        Returns:
            {
                "assignment": "WT_FL_A.2.GLY.H",
                "chain": "WT_FL_A",
                "residue_index": 2,
                "residue_type": "GLY",
                "atom": "H",
            }
        """
        value = PeakList._clean_value(value)

        result = {
            "assignment": None,
            "chain": None,
            "residue_index": None,
            "residue_type": None,
            "atom": None,
        }

        if value is None:
            return result

        assignment = str(value).strip()

        if assignment.startswith("NA:"):
            assignment = assignment[3:]

        result["assignment"] = assignment

        parts = assignment.split(".")

        if len(parts) < 4:
            return result

        chain, residue_index, residue_type = parts[0], parts[1], parts[2]
        atom = ".".join(parts[3:])

        result["chain"] = chain
        result["residue_index"] = PeakList._safe_int(residue_index)
        result["residue_type"] = residue_type
        result["atom"] = atom

        return result

    @staticmethod
    def _ccpnmr_first_existing_column(columns, candidates: list[str]) -> str | None:
        columns = list(columns)
        lookup = {str(col).strip().lower(): col for col in columns}

        for candidate in candidates:
            key = candidate.strip().lower()

            if key in lookup:
                return lookup[key]

        return None

    @staticmethod
    def _is_ccpnmr_dimension_column(column: str) -> bool:
        patterns = [
            r"^Assign F\d+$",
            r"^Assign_F\d+$",
            r"^Pos F\d+$",
            r"^ppmPosition_F\d+$",
            r"^LW F\d+ \(Hz\)$",
            r"^ppmLineWidth_F\d+$",
        ]

        return any(re.match(pattern, column) for pattern in patterns)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        name: str | None = None,
        chemical_shift_columns: Sequence[str] | None = None,
        nuclei: Sequence[str] | None = None,
        nucleus_columns: Sequence[str] | None = None,
        assignment_columns: Sequence[str] | None = None,
        frequency_columns: Sequence[str] | None = None,
        linewidth_columns: Sequence[str] | None = None,
        gaussian_sigma_columns: Sequence[str] | None = None,
        lorentzian_gamma_columns: Sequence[str] | None = None,
        intensity_column: str | None = None,
        volume_column: str | None = None,
        metadata: dict | None = None,
        keep_extra_columns_as_peak_metadata: bool = True,
    ) -> "PeakList":
        """
        Convert a pandas DataFrame into a PeakList.

        Preferred canonical column names:
            chemical_shift_1, chemical_shift_2, ...
            nucleus_1, nucleus_2, ...
            assignment_1, assignment_2, ...
            frequency_1, frequency_2, ...
            linewidth_1, linewidth_2, ...
            gaussian_sigma_1, gaussian_sigma_2, ...
            lorentzian_gamma_1, lorentzian_gamma_2, ...
            intensity
            volume

        For messy Excel files, pass the column names explicitly.
        """
        if chemical_shift_columns is None:
            chemical_shift_columns = cls._infer_dimension_columns(
                df.columns,
                prefixes=("chemical_shift", "shift", "ppm", "position"),
            )

        chemical_shift_columns = list(chemical_shift_columns or [])

        if not chemical_shift_columns:
            raise ValueError(
                "Could not infer chemical-shift columns. "
                "Pass chemical_shift_columns explicitly, e.g. "
                "chemical_shift_columns=['HN', 'N']."
            )

        ndim = len(chemical_shift_columns)

        if nuclei is not None:
            nuclei = list(nuclei)
            if len(nuclei) != ndim:
                raise ValueError("nuclei must have the same length as chemical_shift_columns.")

        nucleus_columns = cls._normalize_optional_dimension_columns(
            nucleus_columns,
            ndim=ndim,
            argument_name="nucleus_columns",
        )
        assignment_columns = cls._normalize_optional_dimension_columns(
            assignment_columns,
            ndim=ndim,
            argument_name="assignment_columns",
            allow_single_shared_column=True,
        )
        frequency_columns = cls._normalize_optional_dimension_columns(
            frequency_columns,
            ndim=ndim,
            argument_name="frequency_columns",
        )
        linewidth_columns = cls._normalize_optional_dimension_columns(
            linewidth_columns,
            ndim=ndim,
            argument_name="linewidth_columns",
        )
        gaussian_sigma_columns = cls._normalize_optional_dimension_columns(
            gaussian_sigma_columns,
            ndim=ndim,
            argument_name="gaussian_sigma_columns",
        )
        lorentzian_gamma_columns = cls._normalize_optional_dimension_columns(
            lorentzian_gamma_columns,
            ndim=ndim,
            argument_name="lorentzian_gamma_columns",
        )

        if intensity_column is None:
            intensity_column = cls._first_existing_column(
                df.columns,
                ["intensity", "height", "peak_height"],
            )

        if volume_column is None:
            volume_column = cls._first_existing_column(
                df.columns,
                ["volume", "integral", "peak_volume"],
            )

        used_columns = set(chemical_shift_columns)
        for columns in [
            nucleus_columns,
            assignment_columns,
            frequency_columns,
            linewidth_columns,
            gaussian_sigma_columns,
            lorentzian_gamma_columns,
        ]:
            if columns:
                used_columns.update(columns)

        if intensity_column:
            used_columns.add(intensity_column)
        if volume_column:
            used_columns.add(volume_column)

        peaks: list[NmrPeak] = []

        for _, row in df.iterrows():
            chemical_shifts = [cls._as_float(row[col]) for col in chemical_shift_columns]

            # Skip empty or incomplete rows.
            if any(value is None for value in chemical_shifts):
                continue

            peak_nuclei = cls._extract_nuclei(
                row=row,
                chemical_shift_columns=chemical_shift_columns,
                nuclei=nuclei,
                nucleus_columns=nucleus_columns,
            )

            assignments = cls._extract_assignments(
                row=row,
                ndim=ndim,
                assignment_columns=assignment_columns,
            )

            frequencies = cls._extract_float_list(row, frequency_columns)
            linewidths = cls._extract_float_list(row, linewidth_columns)
            gaussian_sigmas = cls._extract_float_list(row, gaussian_sigma_columns)
            lorentzian_gammas = cls._extract_float_list(row, lorentzian_gamma_columns)

            intensity = cls._as_float(row[intensity_column]) if intensity_column else None
            volume = cls._as_float(row[volume_column]) if volume_column else None

            peak_metadata = {}
            if keep_extra_columns_as_peak_metadata:
                for col in df.columns:
                    if col not in used_columns:
                        value = cls._clean_value(row[col])
                        if value is not None:
                            peak_metadata[str(col)] = value

            peak = NmrPeak(
                chemical_shifts=chemical_shifts,
                nuclei=peak_nuclei,
                frequencies=frequencies,
                intensity=intensity,
                volume=volume,
                linewidths=linewidths,
                assignments=assignments,
                gaussian_sigmas=gaussian_sigmas,
                lorentzian_gammas=lorentzian_gammas,
                metadata=peak_metadata if peak_metadata else None,
            )

            peaks.append(peak)

        return cls(peaks, name=name, metadata=metadata)

    def to_dataframe(
        self,
        *,
        keep_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Convert the PeakList to a pandas DataFrame.
        """
        rows: list[dict[str, Any]] = []

        for peak in self._peaks:
            row: dict[str, Any] = {}
            ndim = peak.ndim

            # Assignments
            if peak.assignments is not None:
                for dim, assignment in enumerate(peak.assignments, start=1):
                    row[f"assignment_{dim}"] = assignment

            # Chemical shifts
            for dim, chemical_shift in enumerate(peak.chemical_shifts, start=1):
                row[f"chemical_shift_{dim}"] = chemical_shift.ppm

            # Linewidths
            if peak.linewidths is not None:
                for dim, linewidth in enumerate(peak.linewidths, start=1):
                    row[f"linewidth_{dim}"] = linewidth

            # Main scalar properties
            row["intensity"] = peak.intensity

            extra_properties = getattr(peak, "_extra_properties", {}).copy()

            if "S/N" in extra_properties:
                row["S/N"] = extra_properties.pop("S/N")
            elif "S_over_N" in extra_properties:
                row["S/N"] = extra_properties.pop("S_over_N")
            else:
                row["S/N"] = None

            row["volume"] = peak.volume

            # Remove properties already represented explicitly.
            used_keys = {
                "metadata",
                "Height",
                "height",
                "Volume",
                "volume",
                "intensity",
                "S/N",
                "S_over_N",
                "assignments",
                "chemical_shifts",
                "linewidths",
            }

            for key in used_keys:
                extra_properties.pop(key, None)

            # Remaining extra attributes.
            for key, value in extra_properties.items():
                if isinstance(value, (list, tuple)) and len(value) == ndim:
                    for dim, dim_value in enumerate(value, start=1):
                        row[f"{key}_{dim}"] = dim_value
                else:
                    row[key] = value

            rows.append(row)

        df = pd.DataFrame(rows)

        for col in df.columns:
            if col == "residue_index" or col.startswith("residue_index_"):
                df[col] = pd.array(df[col], dtype="Int64")

        # Explicitly enforce final column order.
        preferred_prefixes = [
            "assignment",
            "chemical_shift",
            "linewidth",
        ]

        ordered_columns: list[str] = []

        for prefix in preferred_prefixes:
            ordered_columns.extend(
                sorted(
                    [col for col in df.columns if col.startswith(f"{prefix}_")],
                    key=lambda col: int(col.rsplit("_", 1)[1]),
                )
            )

        for col in ["intensity", "S/N", "volume"]:
            if col in df.columns:
                ordered_columns.append(col)

        remaining_columns = [col for col in df.columns if col not in ordered_columns]

        df = df[ordered_columns + remaining_columns]

        if keep_columns is not None:
            missing_columns = [col for col in keep_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Requested columns are not present in the DataFrame: {missing_columns}")

            df = df[keep_columns]

        if drop_columns is not None:
            missing_columns = [col for col in drop_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Columns requested for dropping are not present in the DataFrame: {missing_columns}")

            df = df.drop(columns=drop_columns)

        return df

    @staticmethod
    def _first_existing_column(columns, candidates: Sequence[str]) -> str | None:
        lookup = {str(col).strip().lower(): col for col in columns}

        for candidate in candidates:
            key = candidate.strip().lower()
            if key in lookup:
                return lookup[key]

        return None

    @staticmethod
    def _normalize_optional_dimension_columns(
        columns: Sequence[str] | None,
        *,
        ndim: int,
        argument_name: str,
        allow_single_shared_column: bool = False,
    ) -> list[str] | None:
        if columns is None:
            return None

        columns = list(columns)

        if allow_single_shared_column and len(columns) == 1:
            return columns

        if len(columns) != ndim:
            raise ValueError(f"{argument_name} must contain {ndim} columns.")

        return columns

    @staticmethod
    def _extract_float_list(row, columns: Sequence[str] | None) -> list[float] | None:
        if columns is None:
            return None

        values = [PeakList._as_float(row[col]) for col in columns]

        if all(value is None for value in values):
            return None

        if any(value is None for value in values):
            raise ValueError(f"Incomplete per-dimension numeric values: {columns}")

        return values

    @classmethod
    def _infer_dimension_columns(
        cls,
        columns,
        *,
        prefixes: Sequence[str],
    ) -> list[str]:
        numbered: list[tuple[int, str]] = []

        for col in columns:
            col_str = str(col).strip()

            for prefix in prefixes:
                pattern = rf"^{re.escape(prefix)}[_\s-]*(\d+)$"
                match = re.match(pattern, col_str, re.IGNORECASE)

                if match:
                    numbered.append((int(match.group(1)), col))
                    break

        if numbered:
            return [col for _, col in sorted(numbered)]

        # Common 2D NMR peak-table aliases.
        lookup = {str(col).strip().lower(): col for col in columns}

        common_sets = [
            ("hn", "n"),
            ("h", "n"),
            ("1h", "15n"),
            ("h_shift", "n_shift"),
            ("c", "h"),
            ("13c", "1h"),
            ("c_shift", "h_shift"),
        ]

        for candidate_set in common_sets:
            if all(candidate in lookup for candidate in candidate_set):
                return [lookup[candidate] for candidate in candidate_set]

        return []

    @classmethod
    def _extract_nuclei(
        cls,
        *,
        row,
        chemical_shift_columns: Sequence[str],
        nuclei: Sequence[str] | None,
        nucleus_columns: Sequence[str] | None,
    ) -> list[str] | None:
        if nuclei is not None:
            return list(nuclei)

        if nucleus_columns is not None:
            values = [cls._clean_value(row[col]) for col in nucleus_columns]

            # Important: do not pass [None, None] to NmrPeak,
            # because NmrPeak would convert None to the string "None".
            if all(value is None for value in values):
                return None

            if any(value is None for value in values):
                raise ValueError("Incomplete nucleus labels.")

            return [str(value) for value in values]

        inferred = [cls._nucleus_from_column_name(col) for col in chemical_shift_columns]

        if all(value is None for value in inferred):
            return None

        if any(value is None for value in inferred):
            # Safer than passing mixed [str, None] into NmrPeak.
            return None

        return inferred

    @classmethod
    def _extract_assignments(
        cls,
        *,
        row,
        ndim: int,
        assignment_columns: Sequence[str] | None,
    ) -> list[str] | None:
        if assignment_columns is None:
            assignments = []

            for dim in range(1, ndim + 1):
                key = f"assignment_{dim}"
                if key in row.index:
                    assignments.append(cls._clean_value(row[key]))
                else:
                    assignments.append(None)

            if any(value is not None for value in assignments):
                return assignments

            if "assignment" in row.index:
                assignment = cls._clean_value(row["assignment"])
                if assignment is not None:
                    return [assignment] * ndim

            return None

        if len(assignment_columns) == 1 and ndim > 1:
            assignment = cls._clean_value(row[assignment_columns[0]])
            return [assignment] * ndim if assignment is not None else None

        assignments = [cls._clean_value(row[col]) for col in assignment_columns]

        if all(value is None for value in assignments):
            return None

        return assignments

    @staticmethod
    def _nucleus_from_column_name(column: str) -> str | None:
        name = str(column).strip().lower()

        if name in {"h", "hn", "1h", "h_shift", "hn_shift"}:
            return "1H"

        if name in {"n", "15n", "n_shift"}:
            return "15N"

        if name in {"c", "13c", "c_shift"}:
            return "13C"

        if name in {"p", "31p", "p_shift"}:
            return "31P"

        return None

    @classmethod
    def from_nef(
        cls,
        file_path: str | Path,
        *,
        name: str | None = None,
        spectrum_index: int = 0,
        metadata: dict | None = None,
    ) -> "PeakList":
        """
        Create a PeakList from a NEF file.

        Requires:
            pip install pynmrstar
        """
        file_path = Path(file_path)
        df = cls._nef_to_dataframe(file_path, spectrum_index=spectrum_index)

        return cls.from_dataframe(
            df,
            name=name or file_path.stem,
            metadata={
                "source": str(file_path),
                "format": "nef",
                "spectrum_index": spectrum_index,
                **(metadata or {}),
            },
        )

    @classmethod
    def _nef_to_dataframe(
        cls,
        file_path: Path,
        *,
        spectrum_index: int = 0,
    ) -> pd.DataFrame:
        import pynmrstar

        entry = pynmrstar.Entry.from_file(file_path)

        try:
            saveframes = entry.get_saveframes_by_category("nef_nmr_spectrum")
        except Exception:
            saveframes = []

        if not saveframes:
            saveframes = [saveframe for saveframe in entry if cls._saveframe_has_loop(saveframe, "_nef_peak_dimension")]

        if not saveframes:
            raise ValueError(f"No NEF peak-list saveframe found in {file_path}")

        if spectrum_index >= len(saveframes):
            raise IndexError(
                f"spectrum_index={spectrum_index} is out of range. Found {len(saveframes)} spectrum saveframe(s)."
            )

        saveframe = saveframes[spectrum_index]

        dim_loop = cls._get_loop(saveframe, "_nef_peak_dimension")
        dim_df = cls._loop_to_dataframe(dim_loop)

        try:
            peak_loop = cls._get_loop(saveframe, "_nef_peak")
            peak_df = cls._loop_to_dataframe(peak_loop)
        except ValueError:
            peak_df = pd.DataFrame()

        peak_id_col = cls._find_tag_column(dim_df, "peak_id")
        dim_id_col = cls._find_tag_column(dim_df, "dimension_id")
        position_col = cls._find_tag_column(dim_df, "position")
        chain_col = cls._find_tag_column(dim_df, "chain_code")
        seq_col = cls._find_tag_column(dim_df, "sequence_code")
        residue_col = cls._find_tag_column(dim_df, "residue_name")
        atom_col = cls._find_tag_column(dim_df, "atom_name")

        if peak_id_col is None or position_col is None:
            raise ValueError("Could not parse NEF peak dimensions. Expected at least peak_id and position columns.")

        peak_info: dict[str, dict[str, Any]] = {}

        if not peak_df.empty:
            peak_peak_id_col = cls._find_tag_column(peak_df, "peak_id", "index")
            height_col = cls._find_tag_column(peak_df, "height", "height_val", "intensity")
            volume_col = cls._find_tag_column(peak_df, "volume", "volume_val")

            if peak_peak_id_col is not None:
                for _, peak_row in peak_df.iterrows():
                    peak_id = str(peak_row[peak_peak_id_col])
                    peak_info[peak_id] = {
                        "intensity": cls._as_float(peak_row[height_col]) if height_col else None,
                        "volume": cls._as_float(peak_row[volume_col]) if volume_col else None,
                    }

        dim_df = dim_df.copy()
        dim_df["_peak_id"] = dim_df[peak_id_col].astype(str)

        rows: list[dict[str, Any]] = []

        for peak_id, group in dim_df.groupby("_peak_id", sort=False):
            if dim_id_col is not None:
                group = group.copy()
                group["_dim_id_numeric"] = group[dim_id_col].map(cls._as_float)
                group = group.sort_values("_dim_id_numeric", na_position="last")

            row: dict[str, Any] = {}

            for dim, (_, dim_row) in enumerate(group.iterrows(), start=1):
                atom_name = cls._clean_value(dim_row[atom_col]) if atom_col else None
                chain_code = cls._clean_value(dim_row[chain_col]) if chain_col else None
                sequence_code = cls._clean_value(dim_row[seq_col]) if seq_col else None
                residue_name = cls._clean_value(dim_row[residue_col]) if residue_col else None

                row[f"chemical_shift_{dim}"] = cls._as_float(dim_row[position_col])
                row[f"nucleus_{dim}"] = cls._nucleus_from_atom_name(atom_name)
                row[f"assignment_{dim}"] = cls._format_nef_assignment(
                    chain_code=chain_code,
                    sequence_code=sequence_code,
                    residue_name=residue_name,
                    atom_name=atom_name,
                )

            row["nef_peak_id"] = peak_id

            if peak_id in peak_info:
                row["intensity"] = peak_info[peak_id].get("intensity")
                row["volume"] = peak_info[peak_id].get("volume")

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _saveframe_has_loop(saveframe, loop_category: str) -> bool:
        target = loop_category.lower()

        for loop in saveframe:
            category = getattr(loop, "category", None)
            if category and category.lower() == target:
                return True

        return False

    @staticmethod
    def _get_loop(saveframe, loop_category: str):
        try:
            return saveframe[loop_category]
        except Exception:
            pass

        target = loop_category.lower()

        for loop in saveframe:
            category = getattr(loop, "category", None)
            if category and category.lower() == target:
                return loop

        raise ValueError(f"Could not find loop {loop_category!r} in saveframe.")

    @staticmethod
    def _loop_to_dataframe(loop) -> pd.DataFrame:
        rows = loop.get_tag(dict_result=True, whole_tag=True)
        return pd.DataFrame(rows)

    @staticmethod
    def _tag_leaf(column: str) -> str:
        return str(column).split(".")[-1].strip().lower()

    @classmethod
    def _find_tag_column(cls, df: pd.DataFrame, *names: str) -> str | None:
        targets = {name.strip().lower() for name in names}

        for col in df.columns:
            if cls._tag_leaf(col) in targets:
                return col

        return None

    @staticmethod
    def _nucleus_from_atom_name(atom_name: str | None) -> str | None:
        if atom_name is None:
            return None

        atom = str(atom_name).strip().upper()

        if atom.startswith("H"):
            return "1H"
        if atom.startswith("N"):
            return "15N"
        if atom.startswith("C"):
            return "13C"
        if atom.startswith("P"):
            return "31P"

        return None

    @staticmethod
    def _format_nef_assignment(
        *,
        chain_code: str | None,
        sequence_code: str | None,
        residue_name: str | None,
        atom_name: str | None,
    ) -> str | None:
        if sequence_code is None and atom_name is None:
            return None

        residue = PeakList._three_to_one(residue_name) if residue_name else ""

        if sequence_code is not None and atom_name is not None:
            assignment = f"{residue}{sequence_code}{atom_name}"
        elif sequence_code is not None:
            assignment = f"{residue}{sequence_code}"
        else:
            assignment = str(atom_name)

        if chain_code:
            return f"{chain_code}:{assignment}"

        return assignment

    @staticmethod
    def _three_to_one(residue_name: str | None) -> str:
        if residue_name is None:
            return ""

        mapping = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
        }

        residue = str(residue_name).strip().upper()

        if len(residue) == 1:
            return residue

        return mapping.get(residue, residue)

    def add(self, peak: NmrPeak) -> None:
        """
        Add a peak to the list.

        Args:
            peak: Peak object to add

        Raises:
            TypeError: If peak is not a Peak object
        """
        if not isinstance(peak, NmrPeak):
            raise TypeError(f"Expected Peak object, got {type(peak)}")
        self._peaks.append(peak)

    def remove(self, peak: NmrPeak) -> None:
        """
        Remove a peak from the list.

        Args:
            peak: Peak object to remove

        Raises:
            ValueError: If peak is not in the list
        """
        self._peaks.remove(peak)

    def remove_at(self, index: int) -> NmrPeak:
        """
        Remove and return peak at specified index.

        Args:
            index: Index of peak to remove

        Returns:
            Removed peak

        Raises:
            IndexError: If index is out of range
        """
        return self._peaks.pop(index)

    def clear(self) -> None:
        """Remove all peaks from the list."""
        self._peaks.clear()

    def extend(self, peaks: Union["PeakList", list[NmrPeak]]) -> None:
        """
        Add multiple peaks to the list.

        Args:
            peaks: PeakList or list of Peak objects
        """
        if isinstance(peaks, PeakList):
            peaks = peaks.peaks
        for peak in peaks:
            self.add(peak)

    def filter_by_dimension(self, ndim: int) -> "PeakList":
        """
        Filter peaks by number of dimensions.

        Args:
            ndim: Number of dimensions

        Returns:
            New PeakList containing only peaks with specified dimensions
        """
        filtered = [peak for peak in self._peaks if peak.ndim == ndim]
        return PeakList(filtered, name=f"{self.name}_dim{ndim}" if self.name else None)

    def filter_by_nucleus(self, nucleus: str, dimension: int | None = None) -> "PeakList":
        """
        Filter peaks by nucleus type.

        Args:
            nucleus: Nucleus label (e.g., '1H', '13C', '15N')
            dimension: If specified, only check this dimension. Otherwise check all dimensions.

        Returns:
            New PeakList containing only peaks with specified nucleus
        """
        filtered = []
        for peak in self._peaks:
            if peak.nuclei is None:
                continue

            if dimension is not None:
                # Check specific dimension
                if dimension < peak.ndim and peak.get_nucleus(dimension) == nucleus:
                    filtered.append(peak)
            else:
                # Check any dimension
                if nucleus in peak.nuclei:
                    filtered.append(peak)

        return PeakList(filtered, name=f"{self.name}_{nucleus}" if self.name else None)

    def filter_by_assignment(self, assignment_pattern: str, dimension: int | None = None) -> "PeakList":
        """
        Filter peaks by assignment pattern.

        Args:
            assignment_pattern: Assignment string or pattern (supports partial matching)
            dimension: If specified, only check this dimension. Otherwise check all dimensions.

        Returns:
            New PeakList containing matching peaks
        """
        filtered = []
        for peak in self._peaks:
            if peak.assignments is None:
                continue

            if dimension is not None:
                # Check specific dimension
                assignment = peak.get_assignment(dimension)
                if assignment and assignment_pattern in assignment:
                    filtered.append(peak)
            else:
                # Check any dimension
                if any(
                    assignment and assignment_pattern in assignment
                    for assignment in peak.assignments
                    if assignment is not None
                ):
                    filtered.append(peak)

        return PeakList(filtered, name=f"{self.name}_{assignment_pattern}" if self.name else None)

    def filter_assigned(self, dimension: int | None = None) -> "PeakList":
        """
        Filter peaks that have assignments.

        Args:
            dimension: If specified, only check this dimension. Otherwise check any dimension.

        Returns:
            New PeakList containing only assigned peaks
        """
        filtered = []
        for peak in self._peaks:
            if peak.assignments is None:
                continue

            if dimension is not None:
                # Check specific dimension
                if peak.get_assignment(dimension) is not None:
                    filtered.append(peak)
            else:
                # Check if any dimension is assigned
                if any(a is not None for a in peak.assignments):
                    filtered.append(peak)

        return PeakList(filtered, name=f"{self.name}_assigned" if self.name else None)

    def filter_unassigned(self) -> "PeakList":
        """
        Filter peaks that have no assignments.

        Returns:
            New PeakList containing only unassigned peaks
        """
        filtered = [
            peak for peak in self._peaks if peak.assignments is None or all(a is None for a in peak.assignments)
        ]
        return PeakList(filtered, name=f"{self.name}_unassigned" if self.name else None)

    def filter_by_shift_range(self, dimension: int, min_ppm: float, max_ppm: float) -> "PeakList":
        """
        Filter peaks by chemical shift range in a specific dimension.

        Args:
            dimension: Dimension index (0-based)
            min_ppm: Minimum chemical shift (ppm)
            max_ppm: Maximum chemical shift (ppm)

        Returns:
            New PeakList containing peaks within the specified range
        """
        filtered = []
        for peak in self._peaks:
            if dimension < peak.ndim:
                shift = peak.get_chemical_shift(dimension).ppm
                if min_ppm <= shift <= max_ppm:
                    filtered.append(peak)

        return PeakList(filtered, name=f"{self.name}_range" if self.name else None)

    def filter_by_intensity(self, min_intensity: float | None = None, max_intensity: float | None = None) -> "PeakList":
        """
        Filter peaks by intensity range.

        Args:
            min_intensity: Minimum intensity (inclusive)
            max_intensity: Maximum intensity (inclusive)

        Returns:
            New PeakList containing peaks within the specified intensity range
        """
        filtered = []
        for peak in self._peaks:
            if peak.intensity is None:
                continue

            if min_intensity is not None and peak.intensity < min_intensity:
                continue
            if max_intensity is not None and peak.intensity > max_intensity:
                continue

            filtered.append(peak)

        return PeakList(filtered, name=f"{self.name}_intensity_filtered" if self.name else None)

    def filter(self, predicate: Callable[[NmrPeak], bool]) -> "PeakList":
        """
        Filter peaks using a custom predicate function.

        Args:
            predicate: Function that takes a Peak and returns True if it should be included

        Returns:
            New PeakList containing peaks that satisfy the predicate

        Example:
            # Filter peaks with intensity > 1000
            high_intensity = peak_list.filter(lambda p: p.intensity and p.intensity > 1000)
        """
        filtered = [peak for peak in self._peaks if predicate(peak)]
        return PeakList(filtered, name=f"{self.name}_filtered" if self.name else None)

    def find_closest(
        self,
        position: list[float],
        max_distance: float | None = None,
        weights: list[float] | None = None,
    ) -> tuple[NmrPeak, float] | None:
        """
        Find the closest peak to a given position.

        Args:
            position: Target position as list of chemical shifts (one per dimension)
            max_distance: Maximum distance to consider (peaks farther away are ignored)
            weights: Optional weights for each dimension in distance calculation

        Returns:
            Tuple of (closest_peak, distance) or None if no peaks found

        Raises:
            ValueError: If peak list is empty
        """
        if not self._peaks:
            return None

        # Create a temporary peak at the target position
        target_peak = NmrPeak(position)

        min_distance = float("inf")
        closest_peak = None

        for peak in self._peaks:
            # Only compare peaks with same dimensions
            if peak.ndim != target_peak.ndim:
                continue

            try:
                distance = peak.distance_to(target_peak, weights=weights)
                if distance < min_distance:
                    if max_distance is None or distance <= max_distance:
                        min_distance = distance
                        closest_peak = peak
            except Exception:
                # Skip peaks that can't be compared (e.g., nucleus mismatch)
                continue

        if closest_peak is None:
            return None

        return closest_peak, min_distance

    def find_within_tolerance(self, position: list[float], tolerances: float | list[float]) -> "PeakList":
        """
        Find all peaks within tolerance of a given position.

        Args:
            position: Target position as list of chemical shifts
            tolerances: Single tolerance or list of tolerances per dimension

        Returns:
            New PeakList containing peaks within tolerance
        """
        target_peak = NmrPeak(position)
        matches = []

        for peak in self._peaks:
            if peak.ndim != target_peak.ndim:
                continue

            try:
                if peak.is_within_tolerance(target_peak, tolerances):
                    matches.append(peak)
            except Exception:
                continue

        return PeakList(matches, name=f"{self.name}_matches" if self.name else None)

    def sort_by_shift(self, dimension: int, reverse: bool = False, inplace: bool = True) -> Optional["PeakList"]:
        """
        Sort peaks by chemical shift in a specific dimension.

        Args:
            dimension: Dimension index (0-based)
            reverse: If True, sort in descending order
            inplace: If True, sort this list. If False, return a new sorted list.

        Returns:
            None if inplace=True, otherwise new sorted PeakList
        """

        def get_shift(peak: NmrPeak) -> float:
            if dimension < peak.ndim:
                return peak.get_chemical_shift(dimension).ppm
            return float("inf")  # Put peaks without this dimension at the end

        if inplace:
            self._peaks.sort(key=get_shift, reverse=reverse)
            return None
        else:
            sorted_peaks = sorted(self._peaks, key=get_shift, reverse=reverse)
            return PeakList(sorted_peaks, name=self.name)

    def sort_by_intensity(self, reverse: bool = True, inplace: bool = True) -> Optional["PeakList"]:
        """
        Sort peaks by intensity.

        Args:
            reverse: If True (default), sort in descending order (highest first)
            inplace: If True, sort this list. If False, return a new sorted list.

        Returns:
            None if inplace=True, otherwise new sorted PeakList
        """

        def get_intensity(peak: NmrPeak) -> float:
            return peak.intensity if peak.intensity is not None else float("-inf")

        if inplace:
            self._peaks.sort(key=get_intensity, reverse=reverse)
            return None
        else:
            sorted_peaks = sorted(self._peaks, key=get_intensity, reverse=reverse)
            return PeakList(sorted_peaks, name=self.name)

    def sort_by_assignment(
        self, dimension: int = 0, reverse: bool = False, inplace: bool = True
    ) -> Optional["PeakList"]:
        """
        Sort peaks by assignment label.

        Args:
            dimension: Dimension index for assignment (0-based)
            reverse: If True, sort in descending order
            inplace: If True, sort this list. If False, return a new sorted list.

        Returns:
            None if inplace=True, otherwise new sorted PeakList
        """

        def get_assignment(peak: NmrPeak) -> str:
            assignment = peak.get_assignment(dimension)
            return assignment if assignment is not None else ""

        if inplace:
            self._peaks.sort(key=get_assignment, reverse=reverse)
            return None
        else:
            sorted_peaks = sorted(self._peaks, key=get_assignment, reverse=reverse)
            return PeakList(sorted_peaks, name=self.name)

    def group_by_assignment(self, dimension: int = 0) -> dict[str, "PeakList"]:
        """
        Group peaks by assignment in a specific dimension.

        Args:
            dimension: Dimension index (0-based)

        Returns:
            Dictionary mapping assignment strings to PeakList objects
        """
        groups = defaultdict(list)

        for peak in self._peaks:
            assignment = peak.get_assignment(dimension)
            key = assignment if assignment is not None else "unassigned"
            groups[key].append(peak)

        return {key: PeakList(peaks, name=f"{self.name}_{key}" if self.name else key) for key, peaks in groups.items()}

    def statistics(self) -> dict:
        """
        Calculate statistics for the peak list.

        Returns:
            Dictionary containing various statistics
        """
        if not self._peaks:
            return {
                "count": 0,
                "dimensions": None,
                "assigned_count": 0,
                "intensities": None,
            }

        intensities = [p.intensity for p in self._peaks if p.intensity is not None]
        assigned_count = sum(1 for p in self._peaks if p.assignments and any(a is not None for a in p.assignments))

        # Count peaks by dimension
        dim_counts = defaultdict(int)
        for peak in self._peaks:
            dim_counts[peak.ndim] += 1

        stats = {
            "count": len(self._peaks),
            "dimensions": dict(dim_counts),
            "assigned_count": assigned_count,
            "unassigned_count": len(self._peaks) - assigned_count,
        }

        if intensities:
            stats["intensities"] = {
                "min": min(intensities),
                "max": max(intensities),
                "mean": np.mean(intensities),
                "median": np.median(intensities),
                "std": np.std(intensities),
            }

        return stats

    def to_dict(self) -> dict:
        """
        Convert peak list to dictionary representation.

        Returns:
            Dictionary containing peak list data
        """
        return {
            "name": self.name,
            "count": len(self._peaks),
            "peaks": [peak.to_dict() for peak in self._peaks],
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PeakList":
        """
        Create PeakList from dictionary representation.

        Args:
            data: Dictionary containing peak list data

        Returns:
            PeakList object
        """
        peaks = [NmrPeak.from_dict(peak_data) for peak_data in data["peaks"]]
        return cls(peaks, name=data.get("name"), metadata=data.get("metadata"))

    def save(self, file_path: str | Path, indent: int = 2) -> None:
        """
        Save peak list to a JSON file.

        Args:
            file_path: Path to the output file
            indent: JSON indentation for readability (default: 2)

        Example:
            peak_list.save('peaks.json')
            peak_list.save('/path/to/peaks.json', indent=4)
        """
        file_path = Path(file_path)

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def load(cls, file_path: str | Path) -> "PeakList":
        """
        Load peak list from a JSON file.

        Args:
            file_path: Path to the input file

        Returns:
            PeakList object

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON

        Example:
            peak_list = PeakList.load('peaks.json')
            peak_list = PeakList.load('/path/to/peaks.json')
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Peak list file not found: {file_path}")

        with open(file_path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __len__(self) -> int:
        """Return number of peaks in the list."""
        return len(self._peaks)

    def __getitem__(self, index: int | slice) -> Union[NmrPeak, "PeakList"]:
        """
        Get peak(s) by index or slice.

        Args:
            index: Integer index or slice

        Returns:
            Peak if index is int, PeakList if index is slice
        """
        if isinstance(index, slice):
            return PeakList(self._peaks[index], name=self.name)
        return self._peaks[index]

    def __setitem__(self, index: int, peak: NmrPeak) -> None:
        """Set peak at index."""
        if not isinstance(peak, NmrPeak):
            raise TypeError(f"Expected Peak object, got {type(peak)}")
        self._peaks[index] = peak

    def __delitem__(self, index: int) -> None:
        """Delete peak at index."""
        del self._peaks[index]

    def __iter__(self) -> Iterator[NmrPeak]:
        """Iterate over peaks."""
        return iter(self._peaks)

    def __contains__(self, peak: NmrPeak) -> bool:
        """Check if peak is in the list."""
        return peak in self._peaks

    def __repr__(self) -> str:
        """Return detailed representation."""

        # Create a defaultdict to count occurrences of each ndim value
        dim_counts = defaultdict(int)

        # Iterate over each peak in the peaklist and count its ndim
        for peak in self._peaks:
            dim_counts[peak.ndim] += 1  # Increment count for each ndim

        # Generate the formatted output (this part sorts the ndim counts)
        dim_info = ", ".join(f"{dim}D: {count}" for dim, count in sorted(dim_counts.items()))

        name_str = f"'{self.name}', " if self.name else ""
        return f"PeakList({name_str}{len(self._peaks)} peaks, {dim_info})"

    def __str__(self) -> str:
        """Return string representation."""
        if not self._peaks:
            return f"PeakList('{self.name}' - empty)" if self.name else "PeakList(empty)"

        name_str = f"'{self.name}' - " if self.name else ""
        return f"PeakList({name_str}{len(self._peaks)} peaks)"
