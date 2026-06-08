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
    """
    Calculate secondary CA-CB chemical-shift deviations.

    For each residue in `df_random_coil`, this function extracts the random-coil
    CA and CB chemical shifts and compares them to the corresponding observed
    CA and CB shifts in `df_cacb`.

    The returned deviation is calculated as:

        (CA_observed - CA_random_coil) - (CB_observed - CB_random_coil)

    Positive values are commonly associated with alpha-helical secondary
    structure propensity, whereas negative values are commonly associated with
    beta-sheet propensity.

    Parameters
    ----------
    df_random_coil:
        DataFrame containing random-coil reference chemical shifts. It must
        contain `index_column`, `"CA"` and `"CB"` columns.

    df_cacb:
        DataFrame, or list of DataFrames, containing observed CA/CB chemical
        shifts. If a list is passed, the DataFrames are concatenated before
        processing.

    atom_column:
        Name of the column in `df_cacb` that identifies the atom type, e.g.
        `"CA"` or `"CB"`.

    index_column:
        Name of the residue-index column shared by `df_random_coil` and
        `df_cacb`.

    chemical_shift_column:
        Name of the column in `df_cacb` containing the observed chemical shift.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns `"residue_index"` and `"cacb_deviation"`.
        Rows for which the random-coil values are non-numeric or for which the
        calculated deviation is missing are skipped.

    Notes
    -----
    If more than one observed CA or CB value is present for a residue, the mean
    value is used.
    """
    if isinstance(df_cacb, pd.DataFrame):
        df_atoms = df_cacb.copy()
    else:
        df_atoms = pd.concat(df_cacb, ignore_index=True)

    df_atoms = df_atoms[df_atoms[index_column].notna()]
    df_atoms = df_atoms.sort_values(by=index_column)

    rows = []
    for _index, row in df_random_coil.iterrows():
        residue_index = row[index_column]
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
    Smooth CA-CB deviations using neighboring values.

    Each value is replaced by a weighted average of itself and its neighboring
    values. The central value receives weight 1.0, and neighboring values receive
    exponentially decreasing weights according to:

        weight = 1 / power**distance

    Missing values are ignored during smoothing.

    Parameters
    ----------
    data:
        Either a DataFrame containing CA-CB deviations, a list of numeric
        values, or a NumPy array.

    index_column:
        Name of the residue-index column. Only used when `data` is a DataFrame.

    deviation_column:
        Name of the column containing CA-CB deviation values. Only used when
        `data` is a DataFrame.

    power:
        Weight-decay factor for neighboring values. Larger values reduce the
        influence of more distant neighbors.

    max_neighbors:
        Maximum number of neighboring positions to include on each side.

    sort_by_index:
        If True, DataFrame input is sorted by `index_column` before smoothing.

    Returns
    -------
    pd.DataFrame | list[float] | np.ndarray
        If `data` is a DataFrame, returns a copy in which `deviation_column`
        contains the smoothed values. The original DataFrame is not modified.

        If `data` is a list, returns a list of smoothed values.

        If `data` is a NumPy array, returns a NumPy array of smoothed values.
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
