"""Build the paper recovery table for the latest adaptive PT-SEM NBA study."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = (
    "skeleton_precision",
    "skeleton_recall",
    "skeleton_f1",
    "directed_precision",
    "directed_recall",
    "directed_f1",
)


def adaptive_rows(
    detail: pd.DataFrame,
    dataset: str,
) -> list[dict[str, object]]:
    game = detail.loc[detail["dataset"].eq(dataset)]
    rows = []
    labels = {"optimization": "PT-SEM (LibraryDP)"}
    for estimator, label in labels.items():
        subset = game.loc[game["estimator"].eq(estimator)]
        n_seasons = int(subset["season"].nunique())
        if n_seasons != 6:
            raise RuntimeError(
                f"{label} has {n_seasons}/6 completed game-quarter seasons"
            )
        row = {"method": label}
        for metric in METRIC_COLUMNS:
            row[metric] = float(subset[metric].mean())
        row["exact_recovery_count"] = int(
            subset["exact_reference_graph"].sum()
        )
        row["n_seasons"] = n_seasons
        rows.append(row)
    return rows


def baseline_rows(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    wanted = (
        "Poisson DAG (ODS-style)",
        "PC (RCIT)",
        "PB-SCM (Cumulant)",
        "PB-SCM (PGF)",
    )
    rows = []
    for method in wanted:
        subset = frame.loc[frame["method"].eq(method)]
        if len(subset) != 1:
            raise RuntimeError(f"missing baseline summary row: {method}")
        source = subset.iloc[0]
        applicable = bool(source.get("model_applicable_all_seasons", True))
        row = {
                "method": method,
                "skeleton_precision": float(source["mean_skeleton_precision"]),
                "skeleton_recall": float(source["mean_skeleton_recall"]),
                "skeleton_f1": float(source["mean_skeleton_f1"]),
                "directed_precision": float(source["mean_directed_precision"]),
                "directed_recall": float(source["mean_directed_recall"]),
                "directed_f1": float(source["mean_directed_f1"]),
                "exact_recovery_count": int(source["exact_recovery_count"]),
                "n_seasons": int(source["n_seasons"]),
                "model_applicable": applicable,
                "applicability_reason": source.get("applicability_reason", ""),
            }
        if not applicable:
            for metric in METRIC_COLUMNS:
                row[metric] = float("nan")
            row["exact_recovery_count"] = pd.NA
        rows.append(row)
    return rows


def metric_text(value: object) -> str:
    return "---" if pd.isna(value) else f"{float(value):.3f}"


def exact_text(count: object, total: object) -> str:
    return "---" if pd.isna(count) else f"{int(count)}/{int(total)}"


def write_markdown(
    frame: pd.DataFrame,
    path: Path,
    caption: str,
) -> None:
    headers = [
        "Method",
        "Skel. P",
        "Skel. R",
        "Skel. F1",
        "Dir. P",
        "Dir. R",
        "Dir. F1",
        "Exact",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * 7) + "|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.method,
                    metric_text(row.skeleton_precision),
                    metric_text(row.skeleton_recall),
                    metric_text(row.skeleton_f1),
                    metric_text(row.directed_precision),
                    metric_text(row.directed_recall),
                    metric_text(row.directed_f1),
                    exact_text(row.exact_recovery_count, row.n_seasons),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            caption,
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def write_latex(
    frame: pd.DataFrame,
    path: Path,
    caption: str,
) -> None:
    latex_caption = caption.replace(
        "FOUL→FTA→FTM",
        r"$\mathrm{FOUL}\to\mathrm{FTA}\to\mathrm{FTM}$",
    ).replace("MISS_FG", r"\texttt{MISS\_FG}")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Skeleton recovery}"
        r" & \multicolumn{3}{c}{Directed recovery} & Exact recovery \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & Precision & Recall & F1 & Precision & Recall & F1 & \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.method)} & "
            f"{metric_text(row.skeleton_precision)} & "
            f"{metric_text(row.skeleton_recall)} & "
            f"{metric_text(row.skeleton_f1)} & "
            f"{metric_text(row.directed_precision)} & "
            f"{metric_text(row.directed_recall)} & "
            f"{metric_text(row.directed_f1)} & "
            f"{exact_text(row.exact_recovery_count, row.n_seasons)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{latex_caption}}}",
            r"\label{tab:nba-adaptive-recovery}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def render_table(
    frame: pd.DataFrame,
    outdir: Path,
    caption: str,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(9.6, 2.55))
    axis.axis("off")
    columns = [
        "Method",
        "Precision",
        "Recall",
        "F1",
        "Precision",
        "Recall",
        "F1",
        "Exact",
    ]
    cells = []
    for row in frame.itertuples(index=False):
        cells.append(
            [
                row.method,
                metric_text(row.skeleton_precision),
                metric_text(row.skeleton_recall),
                metric_text(row.skeleton_f1),
                metric_text(row.directed_precision),
                metric_text(row.directed_recall),
                metric_text(row.directed_f1),
                exact_text(row.exact_recovery_count, row.n_seasons),
            ]
        )
    table = axis.table(
        cellText=cells,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.31, 0.09, 0.09, 0.08, 0.09, 0.09, 0.08, 0.10],
        bbox=[0.015, 0.22, 0.97, 0.62],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.2)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.6 if row in (0, len(cells)) else 0.25)
        if column == 0:
            cell.get_text().set_ha("left")
    axis.text(
        0.385,
        0.91,
        "Skeleton recovery",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
    )
    axis.text(
        0.665,
        0.91,
        "Directed recovery",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
    )
    axis.text(
        0.015,
        0.08,
        textwrap.fill(caption, width=145),
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=8.2,
    )
    for suffix, kwargs in (
        ("pdf", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(
            outdir / f"table_graph_recovery.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/adaptive_ptsem_real_study"),
    )
    parser.add_argument("--dataset", default="five_game")
    parser.add_argument("--stem", default="table_graph_recovery")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path(
            "outputs/nba_five_var_baseline_comparison_fixed/method_summary.csv"
        ),
    )
    args = parser.parse_args()
    root = args.results_dir.resolve()
    detail = pd.read_csv(root / "graph_results_detailed.csv")
    rows = adaptive_rows(detail, args.dataset) + baseline_rows(
        args.baseline_summary.resolve()
    )
    frame = pd.DataFrame(rows)
    if args.dataset == "four_reb_game":
        caption = (
            "Four-variable graph recovery across six seasons relative to "
            "FOUL→FTA→FTM and FTA→REB, after removing MISS_FG. Precision, "
            "recall, and F1 are averaged over seasons; exact recovery "
            "requires the full directed graph. PB-SCM entries are not "
            "reported because the raw counts violate its binomial-thinning "
            "parameter constraints. Undirected PC edges count toward "
            "skeleton recovery only."
        )
    elif args.dataset.startswith("four_"):
        caption = (
            "Four-variable graph recovery across six seasons relative to "
            "FOUL→FTA→FTM, with MISS_FG included as an isolated activity "
            "variable. Precision, recall, and F1 are averaged over seasons; "
            "exact recovery requires the full directed graph. PB-SCM entries "
            "are not reported because the raw counts violate its "
            "binomial-thinning parameter constraints. Undirected PC edges "
            "count toward skeleton recovery only."
        )
    else:
        caption = (
            "Graph recovery relative to the rule-implied NBA event mechanism "
            "across six seasons. Precision, recall, and F1 are averaged over "
            "seasons. Exact recovery counts seasons in which the full directed "
            "graph is recovered exactly."
        )
    frame.to_csv(root / f"{args.stem}.csv", index=False)
    write_markdown(frame, root / f"{args.stem}.md", caption)
    write_latex(frame, root / f"{args.stem}.tex", caption)
    if args.stem != "table_graph_recovery":
        # The renderer uses the canonical basename, then the files are renamed.
        render_table(frame, root, caption)
        for suffix in ("pdf", "png"):
            source = root / f"table_graph_recovery.{suffix}"
            target = root / f"{args.stem}.{suffix}"
            source.replace(target)
    else:
        render_table(frame, root, caption)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
