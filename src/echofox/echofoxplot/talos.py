import pandas as pd
from matplotlib.patches import Ellipse

from ..utils.greek_letters import GreekLetters


def plot_talos_ss(
    ax,
    table: pd.DataFrame,
    *,
    minimum_confidence: float = 0.25,
    color_helix: str = "#e75353",
    pattern_helix: str = r"^[Hh]$",
    color_sheet: str = "#53e0e7",
    pattern_sheet: str = r"^[Ee]$",
    bar_kwargs: dict | None = None,
    ylabel: str = "Talos\nscore",
    xlabel: str | None = "Residue Number",
    draw_legend: bool = True,
    legend_kwargs: dict | None = None,
):
    if bar_kwargs is None:
        bar_kwargs = {
            "width": 1.05,
            "linewidth": 0.5,
            "edgecolor": "black",
        }

    if legend_kwargs is None:
        legend_kwargs = {
            "loc": "lower left",
            "bbox_to_anchor": (0.0, 0.95),
            "ncol": 2,
            "frameon": False,
        }

    ss_list = [
        {"type": "H", "regex_pattern": pattern_helix, "label": "Helix", "color": color_helix},
        {"type": "E", "regex_pattern": pattern_sheet, "label": "Sheet", "color": color_sheet},
    ]

    for ss in ss_list:
        regex_pattern = ss.get("regex_pattern")
        table_filt = table[table["SS_CLASS"].str.match(regex_pattern) & table["CONFIDENCE"].ge(minimum_confidence)]

        color = ss.get("color")

        ax.bar(table_filt["RESID"], table_filt[f"Q_{ss['type']}"], color=color, label=ss.get("label"), **bar_kwargs)

    if xlabel:
        ax.set_xlabel(xlabel, rotation=0, ha="right")

    ax.set_ylabel(ylabel, rotation=0, ha="right")

    if draw_legend:
        ax.legend(**legend_kwargs)
    return


def plot_talos_s2(
    ax,
    table: pd.DataFrame,
    *,
    plot_kwargs: dict | None = None,
    ylabel: str = "RCI S$^2$",
    xlabel: str | None = "Residue Number",
    ylim: tuple = (0, 1),
    draw_legend: bool = True,
    legend_kwargs: dict | None = None,
):
    if plot_kwargs is None:
        plot_kwargs = {
            "marker": "o",
            "linestyle": "-",
        }

    if legend_kwargs is None:
        legend_kwargs = {
            "loc": "lower left",
            "bbox_to_anchor": (0.0, 0.95),
            "ncol": 2,
            "frameon": False,
        }

    ax.plot(
        table["RESID"],
        table["S2"],
        **plot_kwargs,
    )

    if xlabel:
        ax.set_xlabel(xlabel, rotation=0, ha="right")

    ax.set_ylabel(ylabel, rotation=0, ha="right")

    ax.set_ylim(ylim)

    if draw_legend:
        ax.legend(**legend_kwargs)
    return


def plot_talos_chi1_rotamer(
    ax,
    table: pd.DataFrame,
    *,
    gm_color: str = "#e75757",
    t_color: str = "#57e763",
    gp_color: str = "#fcd479",
    ellipse_width: float = 1.0,
    ellipse_max_height: float = 120.0,
    ellipse_kwargs: dict | None = None,
    ylabel: str | None = None,
    xlabel: str | None = "Residue Number",
    ylim: tuple = (-120, 240),
):
    if ellipse_kwargs is None:
        ellipse_kwargs = {
            "alpha": 0.5,
        }

    if ylabel is None:
        ylabel = f"{GreekLetters.Chi}$_1$"

    rotamer_class_map = {
        "g-": {"angle": -60, "color": gm_color, "conf_col": "Q_Gm"},
        "t": {"angle": 60, "color": t_color, "conf_col": "Q_T"},
        "g+": {"angle": 180, "color": gp_color, "conf_col": "Q_Gp"},
    }

    for rotamer_class, rotamer_prop in rotamer_class_map.items():
        table_filt = table[table["CLASS"].eq(rotamer_class)]

        res_idx = table_filt["RESID"].to_numpy()
        confidences = table_filt[rotamer_prop["conf_col"]].to_numpy()

        for res_id, conf in zip(res_idx, confidences):
            ax.add_patch(
                Ellipse(
                    xy=(res_id, rotamer_prop["angle"]),
                    width=ellipse_width,
                    height=ellipse_max_height * conf,
                    color=rotamer_prop["color"],
                    **ellipse_kwargs,
                )
            )

    if xlabel:
        ax.set_xlabel(xlabel, rotation=0, ha="right")

    ax.set_ylabel(ylabel, rotation=0, ha="right")

    ax.set_ylim(ylim)
    return
