import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from echofox.echofoxplot.cacb_deviations import plot_cacb_deviations
from echofox.nmr.cacb_deviations import (
    get_cacb_deviations,
    smooth_cacb_deviations,
)


def test_get_cacb_deviations_calculates_expected_values():
    df_random_coil = pd.DataFrame(
        {
            "residue_index": [1, 2],
            "CA": [55.0, 56.0],
            "CB": [30.0, 31.0],
        }
    )

    df_cacb = pd.DataFrame(
        {
            "residue_index": [1, 1, 2, 2],
            "atom_3": ["CA", "CB", "CA", "CB"],
            "chemical_shift_3": [56.0, 29.0, 58.0, 31.5],
        }
    )

    result = get_cacb_deviations(
        df_random_coil=df_random_coil,
        df_cacb=df_cacb,
        atom_column="atom_3",
        index_column="residue_index",
        chemical_shift_column="chemical_shift_3",
    )

    expected = pd.DataFrame(
        {
            "residue_index": [1, 2],
            "cacb_deviation": [2.0, 1.5],
        }
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected,
        check_dtype=False,
    )


def test_smooth_cacb_deviations_dataframe_replaces_values_without_modifying_original():
    df = pd.DataFrame(
        {
            "residue_index": [1, 2, 3],
            "cacb_deviation": [0.0, 10.0, 20.0],
        }
    )

    result = smooth_cacb_deviations(
        df,
        power=2.0,
        max_neighbors=1,
    )

    assert result["cacb_deviation"].tolist() == pytest.approx([3.3333333, 10.0, 16.6666667])

    assert df["cacb_deviation"].tolist() == [0.0, 10.0, 20.0]


def test_plot_cacb_deviations_accepts_precomputed_deviations():
    fig, ax = plt.subplots()

    cacb_deviations = pd.DataFrame(
        {
            "residue_index": [1, 2, 3],
            "cacb_deviation": [1.0, -2.0, 0.5],
        }
    )

    returned_ax = plot_cacb_deviations(
        ax,
        cacb_deviations=cacb_deviations,
        bar_missing=False,
        bar_zero=False,
        annotate_ylabel=False,
    )

    assert returned_ax is ax
    assert len(ax.patches) == 3

    heights = [patch.get_height() for patch in ax.patches]
    assert heights == pytest.approx([1.0, -2.0, 0.5])

    plt.close(fig)
