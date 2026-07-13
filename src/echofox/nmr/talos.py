from io import StringIO
from pathlib import Path

import pandas as pd


def read_talos_table(
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Read a TALOS-format table into a pandas DataFrame.

    The column names are extracted from the ``VARS`` line. The corresponding
    ``FORMAT`` line is ignored, and the remaining whitespace-separated data
    are loaded into a DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the TALOS table.
    encoding : str, optional
        Character encoding used to read the file. The default is ``"utf-8"``.

    Returns
    -------
    pandas.DataFrame
        Parsed TALOS table.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    IsADirectoryError
        If the specified path points to a directory.
    ValueError
        If the file does not contain valid ``VARS`` and ``FORMAT`` lines or
        contains no table data.
    pandas.errors.ParserError
        If the table data cannot be parsed.
    """
    path = Path(path)

    try:
        lines = path.read_text(encoding=encoding).splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Could not decode TALOS table {path!s} using {encoding!r}.") from exc

    vars_index = next(
        (index for index, line in enumerate(lines) if line.strip().split(maxsplit=1)[0:1] == ["VARS"]),
        None,
    )

    if vars_index is None:
        raise ValueError(f"No VARS line found in TALOS table {path!s}.")

    columns = lines[vars_index].strip().split()[1:]

    if not columns:
        raise ValueError(f"The VARS line in TALOS table {path!s} contains no column names.")

    format_index = next(
        (
            index
            for index in range(vars_index + 1, len(lines))
            if lines[index].strip().split(maxsplit=1)[0:1] == ["FORMAT"]
        ),
        None,
    )

    if format_index is None:
        raise ValueError(f"No FORMAT line found after the VARS line in TALOS table {path!s}.")

    data_lines = [
        line
        for line in lines[format_index + 1 :]
        if line.strip() and not line.lstrip().startswith(("#", "REMARK", "DATA", "VARS", "FORMAT"))
    ]

    if not data_lines:
        raise ValueError(f"No table data found in TALOS table {path!s}.")

    return pd.read_csv(
        StringIO("\n".join(data_lines)),
        sep=r"\s+",
        names=columns,
        header=None,
    )
