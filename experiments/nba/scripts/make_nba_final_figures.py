"""Render the two paper-final NBA figures from the frozen upload package."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle
import numpy as np
import pandas as pd


PACKAGE = Path("outputs/NBA_CHATGPT_UPLOAD_PACKAGE")
RESULTS = PACKAGE / "results"
MIRROR = Path("outputs/nba_extension")
FINAL = Path("outputs/nba_final_colored")
SEASONS = (
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
)
NODES = ("FOUL", "FTA", "FTM", "PERS", "LOOSE")
REFERENCE_EDGES = (
    ("FOUL", "FTA"), ("FTA", "FTM"),
    ("FOUL", "PERS"), ("FOUL", "LOOSE"),
)


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.2,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.035,
})


def save_both(fig: plt.Figure, stem: str) -> None:
    for root in (FINAL, PACKAGE, MIRROR):
        root.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(root / f"{stem}.pdf", format="pdf")
            fig.savefig(root / f"{stem}.png", dpi=300, format="png")
        except PermissionError:
            if root == FINAL:
                raise
            print(f"[skip locked mirror] {root / (stem + '.pdf')}")


def make_figure5() -> None:
    positions = {
        "FOUL": (-1.55, 0.0), "FTA": (0.0, 0.0), "FTM": (1.55, 0.0),
        "PERS": (0.0, 1.0), "LOOSE": (0.0, -1.0),
    }
    radius = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.set_aspect("equal")
    ax.axis("off")

    for source, target in REFERENCE_EDGES:
        arrow = FancyArrowPatch(
            positions[source], positions[target],
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.35, color="#2f4b6e",
            shrinkA=radius * 72, shrinkB=radius * 72,
            connectionstyle="arc3,rad=0",
            zorder=1,
        )
        ax.add_patch(arrow)

    for node, (x, y) in positions.items():
        ax.add_patch(Circle(
            (x, y), radius=radius, facecolor="#e8f0f7",
            edgecolor="#2f4b6e", linewidth=1.15, zorder=2,
        ))
        ax.text(x, y, node, ha="center", va="center",
                fontsize=10.5, fontweight="semibold", zorder=3)

    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-1.48, 1.48)
    save_both(fig, "fig5_rule_implied_nba_graph_final")
    plt.close(fig)


def make_figure6() -> None:
    coef = pd.read_csv(RESULTS / "nba_coefficients_by_season.csv")
    family = pd.read_csv(RESULTS / "nba_working_families_by_season.csv")
    if tuple(coef["season"].drop_duplicates()) != SEASONS:
        raise ValueError("coefficient seasons do not match the ten-season benchmark")
    if tuple(family["season"].drop_duplicates()) != SEASONS:
        raise ValueError("family seasons do not match the ten-season benchmark")
    if len(family) != len(SEASONS) * len(NODES):
        raise ValueError("working-family table is not a complete 5 x 10 grid")

    coef = coef.assign(edge=coef["source"] + "->" + coef["target"])
    edge_order = ("FOUL->FTA", "FTA->FTM", "FOUL->PERS", "FOUL->LOOSE")
    wide = (
        coef[coef["edge"].isin(edge_order)]
        .pivot(index="season", columns="edge", values="coefficient")
        .reindex(SEASONS)
    )
    if not np.isnan(wide.loc["2022-23", "FOUL->LOOSE"]):
        raise ValueError("2022-23 FOUL->LOOSE must remain missing")

    fig, (ax, hx) = plt.subplots(
        1, 2, figsize=(13.6, 5.15),
        gridspec_kw={"width_ratios": (1.02, 1.32), "wspace": 0.20},
    )
    x = np.arange(len(SEASONS))
    styles = {
        "FOUL->FTA": dict(color="#0072b2", marker="o", linestyle="-"),
        "FTA->FTM": dict(color="#d55e00", marker="s", linestyle="--"),
        "FOUL->PERS": dict(color="#009e73", marker="^", linestyle="-."),
        "FOUL->LOOSE": dict(color="#9b6fb6", marker="D", linestyle=":"),
    }
    labels = {
        "FOUL->FTA": r"FOUL$\rightarrow$FTA",
        "FTA->FTM": r"FTA$\rightarrow$FTM",
        "FOUL->PERS": r"FOUL$\rightarrow$PERS",
        "FOUL->LOOSE": r"FOUL$\rightarrow$LOOSE",
    }
    for edge in edge_order:
        ax.plot(
            x, wide[edge].to_numpy(float), label=labels[edge],
            linewidth=1.45, markersize=4.3, markerfacecolor="white",
            markeredgewidth=1.0, **styles[edge],
        )
    ax.set_title("(a) Thinning coefficients", loc="left", pad=8)
    ax.set_ylabel("Estimated thinning coefficient")
    ax.set_xticks(x, SEASONS, rotation=42, ha="right")
    ax.set_xlim(-0.35, len(SEASONS) - 0.65)
    ax.set_ylim(0.0, 1.38)
    ax.set_yticks(np.arange(0.0, 1.41, 0.2))
    ax.grid(axis="y", color="#dddddd", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper left", ncol=2, frameon=False,
        columnspacing=1.15, handlelength=2.5, borderaxespad=0.15,
    )

    family_order = ("Poisson", "NB", "ZIP", "Geom", "Binomial", "Bernoulli")
    short = {
        "Poisson": "Pois", "NB": "NB", "ZIP": "ZIP",
        "Geom": "Geom", "Binomial": "Binom", "Bernoulli": "Bern",
    }
    colors = {
        "Poisson": "#d9e2e8",
        "NB": "#d8c3a5",
        "ZIP": "#c8b7d8",
        "Geom": "#a9c8e8",
        "Binomial": "#8fc8bd",
        "Bernoulli": "#e8b07c",
    }
    fwide = (
        family.pivot(index="node", columns="season", values="family")
        .reindex(index=NODES, columns=SEASONS)
    )
    if fwide.isna().any().any():
        raise ValueError("missing selected working family in node-season grid")
    for i, node in enumerate(NODES):
        for j, season in enumerate(SEASONS):
            hx.add_patch(Rectangle(
                (j - 0.5, i - 0.5), 1.0, 1.0,
                facecolor=colors[fwide.loc[node, season]],
                edgecolor="white", linewidth=0.9,
            ))
    hx.set_title("(b) Selected exogenous family", loc="left", pad=8)
    hx.set_xlim(-0.5, len(SEASONS) - 0.5)
    hx.set_ylim(len(NODES) - 0.5, -0.5)
    hx.set_aspect("auto")
    hx.set_xticks(x, SEASONS, rotation=42, ha="right")
    hx.set_yticks(np.arange(len(NODES)), NODES)
    for i, node in enumerate(NODES):
        for j, season in enumerate(SEASONS):
            fam = fwide.loc[node, season]
            hx.text(j, i, short[fam], ha="center", va="center",
                    fontsize=7.5, color="#111111")
    legend = [
        Patch(facecolor=colors[f], edgecolor="#666666", linewidth=0.4,
              label=short[f]) for f in family_order
    ]
    hx.legend(
        handles=legend, title="Exogenous family", title_fontsize=8.2,
        loc="upper center", bbox_to_anchor=(0.5, -0.25),
        ncol=6, frameon=False, handlelength=1.25, columnspacing=0.9,
    )

    fig.subplots_adjust(left=0.055, right=0.995, top=0.92, bottom=0.27)
    save_both(fig, "fig6_nba_diagnostics_final")
    plt.close(fig)


def main() -> None:
    make_figure5()
    make_figure6()


if __name__ == "__main__":
    main()
