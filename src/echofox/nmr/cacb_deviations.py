from numbers import Real

import numpy as np
import pandas as pd


def get_cacb_deviations(
    df_random_coil: pd.DataFrame,
    df_cacb: pd.DataFrame | list[pd.DataFrame],
    atom_column: str,
    index_column: str,
    chemical_shift_column: str,
):
    if isinstance(df_cacb, pd.DataFrame):
        df_atoms = df_cacb.copy()
    else:
        df_atoms = pd.concat(df_cacb, ignore_index=True)

    df_atoms = df_atoms[df_atoms[index_column].notna()]
    df_atoms = df_atoms.sort_values(by=index_column)

    rows = []
    for _index, row in df_random_coil.iterrows():
        residue_index = residue_index = row[index_column]
        ca_rc = row["CA"]
        cb_rc = row["CB"]

        if not (isinstance(ca_rc, Real) and isinstance(cb_rc, Real) and pd.notna(ca_rc) and pd.notna(cb_rc)):
            continue

        ca = df_atoms.loc[
            (df_atoms[index_column] == residue_index) & (df_atoms[atom_column] == "CA"),
            chemical_shift_column,
        ].mean()

        cb = df_atoms.loc[
            (df_atoms[index_column] == residue_index) & (df_atoms[atom_column] == "CB"),
            chemical_shift_column,
        ].mean()

        cacb_deviation = (ca - ca_rc) - (cb - cb_rc)

        rows.append({"residue_index": residue_index, "cacb_deviation": cacb_deviation})

    cacb_deviations = pd.DataFrame(rows)
    cacb_deviations = cacb_deviations.dropna()

    return cacb_deviations


def smooth_cacb_deviations(
    data: pd.DataFrame | list[float] | np.ndarray,
    *,
    index_column: str = "residue_index",
    deviation_column: str = "cacb_deviation",
    power: float = 2.0,
    max_neighbors: int = 3,
    sort_by_index: bool = True,
):
    """
    Smooth CA-CB deviations using neighboring values with exponentially
    decreasing weights.

    If `data` is a DataFrame, a copy is returned where `deviation_column`
    is replaced by the smoothed values.

    If `data` is a list or NumPy array, the smoothed values are returned in
    the same general format.
    """

    def _smooth_values(values: list[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)

        smoothed_values = []
        weights = [1.0 / (power**i) for i in range(max_neighbors + 1)]

        for i in range(len(values)):
            total = 0.0
            weight_sum = 0.0

            for offset, weight in enumerate(weights):
                signs = [0] if offset == 0 else [-1, 1]

                for sign in signs:
                    idx = i + offset * sign

                    if 0 <= idx < len(values) and pd.notna(values[idx]):
                        total += values[idx] * weight
                        weight_sum += weight

            if weight_sum == 0:
                smoothed_values.append(np.nan)
            else:
                smoothed_values.append(total / weight_sum)

        return np.asarray(smoothed_values)

    if isinstance(data, pd.DataFrame):
        if deviation_column not in data.columns:
            raise ValueError(f"`deviation_column` not found: {deviation_column!r}")

        if sort_by_index and index_column not in data.columns:
            raise ValueError(f"`index_column` not found: {index_column!r}")

        df = data.copy()

        if sort_by_index:
            df = df.sort_values(by=index_column)

        values = pd.to_numeric(df[deviation_column], errors="coerce")
        df[deviation_column] = _smooth_values(values)

        return df

    smoothed = _smooth_values(data)

    if isinstance(data, np.ndarray):
        return smoothed

    return smoothed.tolist()
