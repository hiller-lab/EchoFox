import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba

from echofox.echofoxplot import echofoxplot as efp


def _helix_line(
    start_chain: str,
    start_resi: int,
    end_chain: str,
    end_resi: int,
) -> str:
    line = list(" " * 80)
    line[0:5] = list("HELIX")
    line[19] = start_chain
    line[21:25] = f"{start_resi:>4}"
    line[31] = end_chain
    line[33:37] = f"{end_resi:>4}"
    return "".join(line)


def _sheet_line(
    start_chain: str,
    start_resi: int,
    end_chain: str,
    end_resi: int,
) -> str:
    line = list(" " * 80)
    line[0:5] = list("SHEET")
    line[21] = start_chain
    line[22:26] = f"{start_resi:>4}"
    line[32] = end_chain
    line[33:37] = f"{end_resi:>4}"
    return "".join(line)


def _atom_line(atom_id: int, resi: int, chain_id: str = "A") -> str:
    return (
        f"ATOM  {atom_id:5d}  CA  ALA {chain_id}{resi:4d}    "
        f"{float(resi):8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"  1.00 20.00           C"
    )


@pytest.fixture(autouse=True)
def close_figures() -> None:
    yield
    plt.close("all")


def test_get_secondary_structure_map_returns_expected_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdb_text = "\n".join(
        [
            _helix_line("A", 2, "A", 4),
            _helix_line("A", 6, "A", 7),
            _sheet_line("A", 8, "A", 10),
            *[_atom_line(i, i, "A") for i in range(1, 11)],
            "TER",
            "END",
        ]
    )

    monkeypatch.setitem(
        efp.get_secondary_structure_map.__globals__,
        "_download_pdb_text",
        lambda pdb_code: pdb_text,
    )

    ss_map = efp.get_secondary_structure_map(
        "fake",
        residue_index_offset=10,
        chain_id="A",
    )

    assert ss_map == [
        {"type": "helix", "span": (12, 14)},
        {"type": "helix", "span": (16, 17)},
        {"type": "sheet", "span": (18, 20)},
    ]


def test_draw_secondary_structure_with_user_defined_map() -> None:
    fig, ax = plt.subplots(figsize=(5, 0.3))

    secondary_structure_map = [
        {"type": "helix", "span": (5, 12)},
        {"type": "sheet", "span": (20, 27)},
    ]

    efp.draw_secondary_structure(
        ax,
        secondary_structure_map,
        residue_range=(1, 40),
    )

    assert len(ax.patches) >= 3
    assert ax.get_xlim() == (0.5, 40.5)
    assert ax.get_ylim() == (-0.6, 0.6)
    assert not ax.axison


def test_draw_secondary_structure_with_user_defined_styles() -> None:
    fig, ax = plt.subplots(figsize=(5, 0.3))

    secondary_structure_map = [
        {"type": "helix", "span": (5, 20)},
        {"type": "sheet", "span": (25, 35)},
    ]

    efp.draw_secondary_structure(
        ax,
        secondary_structure_map,
        residue_range=(1, 40),
        unstructured_style={
            "girth": 0.1,
            "color": "lightgrey",
            "edgecolor": "black",
            "linewidth": 0.5,
            "zorder": 1,
        },
        helix_style={
            "height": 0.8,
            "strand_width": 1.2,
            "angle": 35.0,
            "color_primary": "darkred",
            "color_secondary": "salmon",
            "edgecolor": "black",
            "linewidth": 1.5,
            "zorder": 3,
        },
        sheet_style={
            "height": 0.5,
            "arrow_width": 3.0,
            "arrow_height": 0.8,
            "color": "navy",
            "edgecolor": "red",
            "linewidth": 2.0,
            "zorder": 4,
        },
    )

    baseline = ax.patches[0]
    sheet_patch = ax.patches[-1]
    helix_patches = ax.patches[1:-1]

    assert baseline.get_linewidth() == pytest.approx(0.1)
    assert baseline.get_facecolor() == to_rgba("lightgrey")
    assert baseline.get_edgecolor() == to_rgba("black")

    assert len(helix_patches) > 0
    assert any(patch.get_facecolor() == to_rgba("darkred") for patch in helix_patches)
    assert any(patch.get_facecolor() == to_rgba("salmon") for patch in helix_patches)
    assert all(patch.get_edgecolor() == to_rgba("black") for patch in helix_patches)
    assert all(patch.get_linewidth() == pytest.approx(1.5) for patch in helix_patches)

    assert sheet_patch.get_facecolor() == to_rgba("navy")
    assert sheet_patch.get_edgecolor() == to_rgba("red")
    assert sheet_patch.get_linewidth() == pytest.approx(2.0)


def test_draw_secondary_structure_with_domain_color_map() -> None:
    fig, ax = plt.subplots(figsize=(12, 0.3))

    secondary_structure_map = [
        {"span": (4, 10), "type": "helix"},
        {"span": (166, 170), "type": "sheet"},
    ]

    domain_color_map = [
        {"name": "JD", "range": (0, 78), "color": "#d23d3d"},
        {"name": "CTD1", "range": (159, 246), "color": "#68a7ce"},
    ]

    efp.draw_secondary_structure(
        ax,
        secondary_structure_map,
        residue_range=(1, 340),
        domain_color_map=domain_color_map,
        unstructured_style={"girth": 0.1},
    )

    sheet_patch = ax.patches[-1]

    assert len(ax.patches) >= 3
    assert sheet_patch.get_facecolor() == to_rgba("#68a7ce")
