from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, TypeAlias

try:
    from typing import NotRequired, TypedDict
except ImportError:
    from typing_extensions import NotRequired, TypedDict

from io import StringIO
from typing import TypedDict
from urllib.request import urlopen

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
from Bio.PDB import PDBParser
from matplotlib.axes import Axes

from echofox.core.colors import Color

Number: TypeAlias = int | float
ResidueSpan: TypeAlias = tuple[Number, Number]
SecondaryStructureType: TypeAlias = Literal["helix", "sheet"]


class SecondaryStructureSpan(TypedDict):
    type: SecondaryStructureType
    span: tuple[int, int]


class SecondaryStructureElement(TypedDict):
    type: SecondaryStructureType
    span: ResidueSpan
    color: NotRequired[str]


class DomainColorElement(TypedDict):
    range: ResidueSpan
    color: str
    name: NotRequired[str]


def _download_pdb_text(pdb_code: str) -> str:
    """Download a legacy PDB file from RCSB into memory."""
    pdb_code = pdb_code.lower().strip()

    if not pdb_code:
        raise ValueError("pdb_code must not be empty.")

    url = f"https://files.rcsb.org/download/{pdb_code}.pdb"

    with urlopen(url) as response:
        return response.read().decode("utf-8")


def get_secondary_structure_map(
    pdb_code: str,
    residue_index_offset: int = 0,
    chain_id: str | None = None,
) -> list[SecondaryStructureSpan]:
    """
    Get helix/sheet residue spans for one chain from the HELIX/SHEET records
    of a legacy PDB file.

    Parameters
    ----------
    pdb_code:
        PDB accession code, e.g. "1ubq".
    residue_index_offset:
        Value added to residue numbers in the returned spans.
    chain_id:
        Chain to analyze. If None, the first chain in model 0 is used.

    Returns
    -------
    list[SecondaryStructureSpan]
        Example:
        [
            {"type": "helix", "span": (12, 24)},
            {"type": "sheet", "span": (45, 51)},
        ]
    """
    pdb_code = pdb_code.lower().strip()
    pdb_text = _download_pdb_text(pdb_code)

    # Parse structure to get first chain ID
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_code, StringIO(pdb_text))

    if structure is None:
        raise ValueError(f"Could not parse structure for PDB code {pdb_code!r}.")

    model = structure[0]

    if chain_id is None:
        try:
            chain_id = next(model.get_chains()).id
        except StopIteration as exc:
            raise ValueError(f"No chains found in PDB entry {pdb_code!r}.") from exc

    # Collect residue-level SS info
    ss_info: dict[int, SecondaryStructureType] = {}

    for line in pdb_text.splitlines():
        if line.startswith("HELIX"):
            start_chain_id = line[19].strip()
            end_chain_id = line[31].strip()

            if start_chain_id == chain_id and end_chain_id == chain_id:
                start_resi = int(line[21:25].strip())
                end_resi = int(line[33:37].strip())

                for resi in range(start_resi, end_resi + 1):
                    ss_info[resi] = "helix"

        elif line.startswith("SHEET"):
            start_chain_id = line[21].strip()
            end_chain_id = line[32].strip()

            if start_chain_id == chain_id and end_chain_id == chain_id:
                start_resi = int(line[22:26].strip())
                end_resi = int(line[33:37].strip())

                for resi in range(start_resi, end_resi + 1):
                    ss_info[resi] = "sheet"

    # Merge consecutive residues into spans
    secondary_structure_map: list[SecondaryStructureSpan] = []
    current_type: SecondaryStructureType | None = None
    current_start: int | None = None
    previous_resi: int | None = None

    for resi in sorted(ss_info):
        ss_type = ss_info[resi]

        starts_new_span = current_type != ss_type or previous_resi is None or resi != previous_resi + 1

        if starts_new_span:
            if current_type is not None and current_start is not None and previous_resi is not None:
                secondary_structure_map.append(
                    {
                        "type": current_type,
                        "span": (
                            current_start + residue_index_offset,
                            previous_resi + residue_index_offset,
                        ),
                    }
                )

            current_type = ss_type
            current_start = resi

        previous_resi = resi

    # Append last span
    if current_type is not None and current_start is not None and previous_resi is not None:
        secondary_structure_map.append(
            {
                "type": current_type,
                "span": (
                    current_start + residue_index_offset,
                    previous_resi + residue_index_offset,
                ),
            }
        )

    return secondary_structure_map


def _clip_span_to_range(
    span: ResidueSpan,
    residue_range: ResidueSpan,
) -> tuple[float, float] | None:
    """Clip a secondary-structure span to the visible residue range."""
    span_start, span_end = _normalize_span(span)
    range_start, range_end = _normalize_span(residue_range)

    clipped_start = max(span_start, range_start)
    clipped_end = min(span_end, range_end)

    if clipped_start > clipped_end:
        return None

    return clipped_start, clipped_end


def _find_domain_color(
    span: ResidueSpan,
    domain_color_map: Sequence[DomainColorElement] | None,
) -> str | None:
    """Return domain color if span lies fully inside one domain."""
    if domain_color_map is None:
        return None

    span_start, span_end = _normalize_span(span)

    for domain in domain_color_map:
        domain_start, domain_end = _normalize_span(domain["range"])

        if domain_start <= span_start and span_end <= domain_end:
            return domain["color"]

    return None


def _expanded_residue_span(span: ResidueSpan) -> tuple[float, float]:
    """
    Convert residue-number span to drawing coordinates.

    Residue 10 is drawn from 9.5 to 10.5.
    Span (10, 20) is drawn from 9.5 to 20.5.
    """
    start, end = _normalize_span(span)
    return start - 0.5, end + 0.5


def _draw_rect(
    target_ax: Axes,
    girth: float,
    span: ResidueSpan,
    color: str,
    edgecolor: str | None = None,
    zorder: int = 1,
) -> None:
    """Draw the unstructured baseline."""
    x_start, x_end = _expanded_residue_span(span)
    width = x_end - x_start

    if width <= 0:
        return

    rect = mpatches.Rectangle(
        (x_start, 0),
        width,
        0.0,
        facecolor=color,
        edgecolor=edgecolor if edgecolor is not None else color,
        linewidth=girth,
        clip_on=False,
        zorder=zorder,
    )

    target_ax.add_patch(rect)


def _draw_helix(
    ax: Axes,
    span: ResidueSpan,
    height: float,
    annot: str = "",
    strand_width: float = 1.0,
    angle: float = 35.0,
    colors: tuple[str, str] = ("#dc143c", "#ffc0cb"),
    edgecolor: str = "black",
    linewidth: float = 0.5,
    zorder: int = 3,
) -> None:
    """Draw a stylized alpha helix as alternating diagonal ribbons."""
    x_start, x_end = _expanded_residue_span(span)
    width = x_end - x_start

    if width <= 0:
        return

    if not 0 < angle < 90:
        raise ValueError("angle must be between 0 and 90 degrees.")

    if strand_width <= 0:
        raise ValueError("strand_width must be positive.")

    diagonal_length = height / np.sin(np.deg2rad(angle))
    diagonal_dx = diagonal_length * np.cos(np.deg2rad(angle))

    step = 2 * diagonal_dx + 0.2
    offset = 0.1

    # Fallback for very short helices.
    if width < max(strand_width + diagonal_dx, 1.0):
        rect = mpatches.Rectangle(
            (x_start, -height / 2),
            width,
            height,
            facecolor=colors[0],
            edgecolor=edgecolor,
            linewidth=linewidth,
            clip_on=True,
            zorder=zorder,
        )
        ax.add_patch(rect)
    else:
        # Backward diagonal strands.
        current_x = x_start + diagonal_dx + offset

        while current_x + strand_width + diagonal_dx < x_end:
            codes, verts = zip(
                *[
                    (mpath.Path.MOVETO, (current_x, height / 2)),
                    (mpath.Path.LINETO, (current_x + diagonal_dx, -height / 2)),
                    (
                        mpath.Path.LINETO,
                        (current_x + diagonal_dx + strand_width, -height / 2),
                    ),
                    (mpath.Path.LINETO, (current_x + strand_width, height / 2)),
                    (mpath.Path.CLOSEPOLY, (current_x, height / 2)),
                ]
            )

            patch = mpatches.PathPatch(
                mpath.Path(verts, codes),
                facecolor=colors[1],
                edgecolor=edgecolor,
                linewidth=linewidth,
                clip_on=True,
                zorder=zorder,
            )
            ax.add_patch(patch)

            current_x += step

        # Forward diagonal strands.
        current_x = x_start

        while current_x + strand_width + diagonal_dx < x_end:
            codes, verts = zip(
                *[
                    (mpath.Path.MOVETO, (current_x, -height / 2)),
                    (mpath.Path.LINETO, (current_x + diagonal_dx, height / 2)),
                    (
                        mpath.Path.LINETO,
                        (current_x + diagonal_dx + strand_width, height / 2),
                    ),
                    (mpath.Path.LINETO, (current_x + strand_width, -height / 2)),
                    (mpath.Path.CLOSEPOLY, (current_x, -height / 2)),
                ]
            )

            patch = mpatches.PathPatch(
                mpath.Path(verts, codes),
                facecolor=colors[0],
                edgecolor=edgecolor,
                linewidth=linewidth,
                clip_on=True,
                zorder=zorder,
            )
            ax.add_patch(patch)

            current_x += step

    if annot:
        ax.text(
            x_start + width / 2,
            -0.75,
            annot,
            ha="center",
            va="top",
            zorder=zorder + 1,
        )


def _draw_sheet(
    ax: Axes,
    span: ResidueSpan,
    height: float,
    annot: str = "",
    arrow_width: float = 2.0,
    arrow_height: float = 1.0,
    color: str = "#1e90ff",
    edgecolor: str = "black",
    linewidth: float = 0.5,
    zorder: int = 3,
) -> None:
    """Draw a beta strand as an arrow."""
    x_start, x_end = _expanded_residue_span(span)
    width = x_end - x_start

    if width <= 0:
        return

    # Prevent the arrowhead from becoming larger than the whole strand.
    arrow_head_width = min(max(arrow_width, width * 0.1), width)
    arrow_head_height = arrow_height
    rectangle_width = max(width - arrow_head_width, 0.0)

    codes, verts = zip(
        *[
            (mpath.Path.MOVETO, (x_start, height / 2)),
            (mpath.Path.LINETO, (x_start + rectangle_width, height / 2)),
            (
                mpath.Path.LINETO,
                (x_start + rectangle_width, arrow_head_height / 2),
            ),
            (mpath.Path.LINETO, (x_end, 0.0)),
            (
                mpath.Path.LINETO,
                (x_start + rectangle_width, -arrow_head_height / 2),
            ),
            (mpath.Path.LINETO, (x_start + rectangle_width, -height / 2)),
            (mpath.Path.LINETO, (x_start, -height / 2)),
            (mpath.Path.CLOSEPOLY, (x_start, height / 2)),
        ]
    )

    patch = mpatches.PathPatch(
        mpath.Path(verts, codes),
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        clip_on=True,
        zorder=zorder,
    )

    ax.add_patch(patch)

    if annot:
        ax.text(
            x_start + width / 2,
            -0.75,
            annot,
            ha="center",
            va="top",
            zorder=zorder + 1,
        )


def _as_axes_list(axs: Axes | Iterable[Axes]) -> list[Axes]:
    """Normalize a single Axes or iterable/array of Axes to a flat list."""
    if isinstance(axs, Axes):
        return [axs]

    if isinstance(axs, np.ndarray):
        return list(axs.ravel())

    return list(axs)


def _normalize_span(span: ResidueSpan) -> tuple[float, float]:
    """Return span as ordered floats."""
    start, end = float(span[0]), float(span[1])
    return (start, end) if start <= end else (end, start)


DEFAULT_UNSTRUCTURED_STYLE = {
    "girth": 1.5,
    "color": "#b3b3b3",
    "edgecolor": None,
    "zorder": 1,
}

DEFAULT_HELIX_STYLE = {
    "height": 1.0,
    "strand_width": 1.5,
    "angle": 35.0,
    "color_primary": "#dc143c",
    "color_secondary": "#ffc0cb",
    "edgecolor": "black",
    "linewidth": 1.0,
    "zorder": 3,
}

DEFAULT_SHEET_STYLE = {
    "height": 0.65,
    "arrow_width": 2.0,
    "arrow_height": 1.0,
    "color": "#1e90ff",
    "edgecolor": "black",
    "linewidth": 1.0,
    "zorder": 3,
}


def draw_secondary_structure(
    axs: Axes | Iterable[Axes],
    secondary_structure_map: str | Sequence[SecondaryStructureElement],
    residue_range: ResidueSpan,
    domain_color_map: Sequence[DomainColorElement] | None = None,
    edgewidth: int | float = 1.0,
    unstructured_style: dict | None = None,
    helix_style: dict | None = None,
    sheet_style: dict | None = None,
) -> None:
    """
    Draw a secondary-structure cartoon onto one or more Matplotlib axes.

    The cartoon shows unstructured regions as a thin baseline, helices as
    stylized diagonal ribbons, and beta strands as arrows. Secondary-structure
    elements can either be supplied directly as a list of dictionaries or
    generated from a PDB code via `get_secondary_structure_map()`.

    Parameters
    ----------
    axs:
        One Matplotlib Axes object or an iterable/array of Axes objects.

    secondary_structure_map:
        Either a PDB code, for example `"1ubq"`, or a sequence of secondary-
        structure elements.

        Each element must have the form:

        {
            "type": "helix" | "sheet",
            "span": tuple[int | float, int | float],
        }

        Optionally, an element may also contain:

        {
            "color": str,
        }

    residue_range:
        Visible residue range, for example `(1, 120)`.

        This range defines the full horizontal extent of the cartoon, including
        unstructured N- and C-terminal regions. It is always required, because
        secondary-structure annotations only describe helices and sheets and
        therefore do not reliably define the full protein length.

    domain_color_map:
        Optional domain color mapping. A secondary-structure element is colored
        by a domain if its span lies fully inside the domain range.

        Example:

        [
            {"range": (1, 50), "color": "#1f77b4"},
            {"range": (51, 120), "color": "#ff7f0e"},
        ]

    edgewidth:
        Default outline width for helices and sheets. This is kept for
        convenience. It is overridden by `helix_style["linewidth"]` or
        `sheet_style["linewidth"]` if those keys are given.

    unstructured_style:
        Optional style overrides for the unstructured baseline.

        Available keys:

        {
            "girth": float,
            "color": str,
            "edgecolor": str | None,
            "zorder": int,
        }

    helix_style:
        Optional style overrides for helices.

        Available keys:

        {
            "height": float,
            "strand_width": float,
            "angle": float,
            "color_primary": str,
            "color_secondary": str,
            "edgecolor": str,
            "linewidth": float,
            "zorder": int,
        }

    sheet_style:
        Optional style overrides for beta strands.

        Available keys:

        {
            "height": float,
            "arrow_width": float,
            "arrow_height": float,
            "color": str,
            "edgecolor": str,
            "linewidth": float,
            "zorder": int,
        }

    Raises
    ------
    ValueError
        If an unknown secondary-structure type is encountered.
    """
    axes = _as_axes_list(axs)

    if isinstance(secondary_structure_map, str):
        secondary_structure_map = get_secondary_structure_map(secondary_structure_map)

    range_start, range_end = _normalize_span(residue_range)

    # Merge defaults with user-provided styles.
    unstructured = {
        **DEFAULT_UNSTRUCTURED_STYLE,
        **(unstructured_style or {}),
    }

    helix = {
        **DEFAULT_HELIX_STYLE,
        "linewidth": float(edgewidth),
        **(helix_style or {}),
    }

    sheet = {
        **DEFAULT_SHEET_STYLE,
        "linewidth": float(edgewidth),
        **(sheet_style or {}),
    }

    for ax in axes:
        ax.set_xlim(range_start - 0.5, range_end + 0.5)
        ax.set_ylim(-0.6, 0.6)

        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        ax.set_axis_off()

        _draw_rect(
            ax,
            girth=unstructured["girth"],
            span=residue_range,
            color=unstructured["color"],
            edgecolor=unstructured["edgecolor"],
            zorder=unstructured["zorder"],
        )

        for secondary_structure in secondary_structure_map:
            clipped_span = _clip_span_to_range(
                secondary_structure["span"],
                residue_range,
            )

            if clipped_span is None:
                continue

            domain_color = _find_domain_color(
                secondary_structure["span"],
                domain_color_map,
            )

            ss_color = secondary_structure.get("color", domain_color)

            if secondary_structure["type"] == "sheet":
                _draw_sheet(
                    ax,
                    clipped_span,
                    height=sheet["height"],
                    annot="",
                    arrow_width=sheet["arrow_width"],
                    arrow_height=sheet["arrow_height"],
                    color=ss_color if ss_color is not None else sheet["color"],
                    edgecolor=sheet["edgecolor"],
                    linewidth=sheet["linewidth"],
                    zorder=sheet["zorder"],
                )

            elif secondary_structure["type"] == "helix":
                if ss_color is not None:
                    helix_colors = (
                        ss_color,
                        Color(ss_color).adjust_hsv(sat_mult=0.27, val_mult=1.58).hex,
                    )
                else:
                    helix_colors = (
                        helix["color_primary"],
                        helix["color_secondary"],
                    )

                _draw_helix(
                    ax,
                    clipped_span,
                    height=helix["height"],
                    annot="",
                    strand_width=helix["strand_width"],
                    angle=helix["angle"],
                    colors=helix_colors,
                    edgecolor=helix["edgecolor"],
                    linewidth=helix["linewidth"],
                    zorder=helix["zorder"],
                )

            else:
                raise ValueError(f"Unknown secondary-structure type: {secondary_structure['type']!r}")
