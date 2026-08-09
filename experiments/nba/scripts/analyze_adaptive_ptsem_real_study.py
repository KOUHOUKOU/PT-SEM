"""Summarize and audit the adaptive-family NBA PT-SEM results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REFERENCE_EDGES = {
    "three_team": {
        ("FOUL", "FTA"),
        ("FTA", "FTM"),
    },
    "five_team": {
        ("FOUL", "FTA"),
        ("FTA", "FTM"),
        ("FTA", "REB"),
        ("MISS_FG", "REB"),
    },
    "four_team": {
        ("FOUL", "FTA"),
        ("FTA", "FTM"),
    },
    "five_game": {
        ("FOUL", "FTA"),
        ("FTA", "FTM"),
        ("FTA", "REB"),
        ("MISS_FG", "REB"),
    },
    "four_game": {
        ("FOUL", "FTA"),
        ("FTA", "FTM"),
    },
}


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def parse_edges(text: str) -> set[tuple[str, str]]:
    if not text or text == "(none)":
        return set()
    edges = set()
    for part in text.split(","):
        left, right = part.strip().split("->", 1)
        edges.add((left.strip(), right.strip()))
    return edges


def format_edges(edges: set[tuple[str, str]]) -> str:
    return ", ".join(f"{a}->{b}" for a, b in sorted(edges)) or "(none)"


def score_graph(
    predicted: set[tuple[str, str]],
    reference: set[tuple[str, str]],
) -> dict[str, Any]:
    true_positive = len(predicted & reference)
    false_positive = len(predicted - reference)
    false_negative = len(reference - predicted)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    predicted_skeleton = {
        frozenset((source, target)) for source, target in predicted
    }
    reference_skeleton = {
        frozenset((source, target)) for source, target in reference
    }
    skeleton_true_positive = len(predicted_skeleton & reference_skeleton)
    skeleton_false_positive = len(predicted_skeleton - reference_skeleton)
    skeleton_false_negative = len(reference_skeleton - predicted_skeleton)
    skeleton_precision = (
        skeleton_true_positive
        / (skeleton_true_positive + skeleton_false_positive)
        if skeleton_true_positive + skeleton_false_positive
        else 0.0
    )
    skeleton_recall = (
        skeleton_true_positive
        / (skeleton_true_positive + skeleton_false_negative)
        if skeleton_true_positive + skeleton_false_negative
        else 0.0
    )
    skeleton_f1 = (
        2.0
        * skeleton_precision
        * skeleton_recall
        / (skeleton_precision + skeleton_recall)
        if skeleton_precision + skeleton_recall
        else 0.0
    )
    return {
        "skeleton_precision": skeleton_precision,
        "skeleton_recall": skeleton_recall,
        "skeleton_f1": skeleton_f1,
        "skeleton_tp": skeleton_true_positive,
        "skeleton_fp": skeleton_false_positive,
        "skeleton_fn": skeleton_false_negative,
        "directed_precision": precision,
        "directed_recall": recall,
        "directed_f1": f1,
        "directed_tp": true_positive,
        "directed_fp": false_positive,
        "directed_fn": false_negative,
        "exact_reference_graph": predicted == reference,
        "core_foul_chain": {
            ("FOUL", "FTA"),
            ("FTA", "FTM"),
        }.issubset(predicted),
    }


def load_graph_results(root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(root.glob("*/*/*/graph_result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        results.append(payload)
    if not results:
        raise FileNotFoundError(f"no graph_result.json files under {root}")
    return results


def graph_detail_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        predicted = parse_edges(result["edges"])
        metrics = score_graph(predicted, REFERENCE_EDGES[result["dataset"]])
        rows.append(
            {
                "dataset": result["dataset"],
                "unit": result["unit"],
                "season": result["season"],
                "estimator": result["estimator"],
                "n": result["n"],
                "edges": result["edges"],
                "n_edges": len(predicted),
                "selected_families": ";".join(result["selected_families"]),
                "score": result["score"],
                "bic_delta_second": result["bic_delta_second"],
                "runtime_sec": result["runtime_sec"],
                "reference_edges": format_edges(
                    REFERENCE_EDGES[result["dataset"]]
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "estimator", "season"]
    )


def edge_stability_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for (dataset, estimator), group in itertools_group(
        results, ("dataset", "estimator")
    ):
        variables = group[0]["variables"]
        edge_sets = [parse_edges(item["edges"]) for item in group]
        reference = REFERENCE_EDGES[dataset]
        for source in variables:
            for target in variables:
                if source == target:
                    continue
                count = sum((source, target) in edges for edges in edge_sets)
                rows.append(
                    {
                        "dataset": dataset,
                        "estimator": estimator,
                        "source": source,
                        "target": target,
                        "edge": f"{source}->{target}",
                        "selected_seasons": count,
                        "n_seasons": len(group),
                        "selection_rate": count / len(group),
                        "in_reference_graph": (source, target) in reference,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "estimator", "selection_rate", "edge"],
        ascending=[True, True, False, True],
    )


def itertools_group(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
):
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    for key in sorted(groups):
        yield key, groups[key]


def family_selection_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        for node_fit in result["node_fits"]:
            params = node_fit["params"]
            boundary = False
            if node_fit["family"] == "Poisson":
                boundary = float(params.get("lam", 1.0)) < 1e-6
            elif node_fit["family"] == "NB":
                boundary = (
                    float(params.get("p", 0.5)) > 1.0 - 1e-6
                    or float(params.get("r", 1.0)) > 1e7
                )
            elif node_fit["family"] == "ZIP":
                boundary = float(params.get("rho", 0.5)) < 1e-6
            elif node_fit["family"] == "Geom":
                boundary = float(params.get("p", 0.5)) > 1.0 - 1e-6
            elif node_fit["family"] in ("Binomial", "Bernoulli"):
                probability = float(params.get("p", 0.5))
                boundary = probability < 1e-6 or probability > 1.0 - 1e-6
            rows.append(
                {
                    "dataset": result["dataset"],
                    "unit": result["unit"],
                    "season": result["season"],
                    "estimator": result["estimator"],
                    "node": node_fit["node"],
                    "parent_set": ";".join(node_fit["parent_set"]),
                    "family": node_fit["family"],
                    "params": json.dumps(params, sort_keys=True),
                    "alpha": json.dumps(node_fit["alpha"], sort_keys=True),
                    "local_bic": node_fit["local_bic"],
                    "local_loglik": node_fit["local_loglik"],
                    "optimizer_success": node_fit["optimizer_success"],
                    "optimizer_status": node_fit["optimizer_status"],
                    "boundary_solution": boundary,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "estimator", "season", "node"]
    )


def estimator_comparison_frame(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "season"]
    for key, group in detail.groupby(keys):
        by_method = group.set_index("estimator")
        if not {"moment", "optimization"}.issubset(by_method.index):
            continue
        moment = by_method.loc["moment"]
        optimized = by_method.loc["optimization"]
        rows.append(
            {
                "dataset": key[0],
                "season": key[1],
                "same_graph": moment["edges"] == optimized["edges"],
                "moment_edges": moment["edges"],
                "optimization_edges": optimized["edges"],
                "moment_core_chain": moment["core_foul_chain"],
                "optimization_core_chain": optimized["core_foul_chain"],
                "moment_directed_f1": moment["directed_f1"],
                "optimization_directed_f1": optimized["directed_f1"],
                "moment_score": moment["score"],
                "optimization_score": optimized["score"],
                "score_improvement": moment["score"] - optimized["score"],
            }
        )
    return pd.DataFrame(rows)


def optimization_diagnostics(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*/*/optimization/local_family_fits.csv")):
        frame = pd.read_csv(path)
        finite = np.isfinite(frame["bic"].to_numpy(float))
        improvements = frame.loc[finite, "loglik_improvement"].to_numpy(float)
        rows.append(
            {
                "dataset": frame["dataset"].iloc[0],
                "season": frame["season"].iloc[0],
                "n_local_family_fits": len(frame),
                "n_finite": int(finite.sum()),
                "n_optimizer_success": int(frame["success"].fillna(False).sum()),
                "n_moment_start_retained": int(
                    frame["status"].eq("moment_start_retained").sum()
                ),
                "minimum_loglik_improvement": (
                    float(np.min(improvements)) if len(improvements) else math.nan
                ),
                "median_loglik_improvement": (
                    float(np.median(improvements)) if len(improvements) else math.nan
                ),
                "maximum_loglik_improvement": (
                    float(np.max(improvements)) if len(improvements) else math.nan
                ),
                "total_local_runtime_sec": float(frame["runtime_sec"].sum()),
                "total_function_evaluations": int(frame["nfev"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "season"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_final_audit(
    root: Path,
    results: list[dict[str, Any]],
    diagnostics: pd.DataFrame,
) -> None:
    failures = []
    observed_datasets = sorted({item["dataset"] for item in results})
    observed_estimators = sorted({item["estimator"] for item in results})
    expected_cells = len(observed_datasets) * len(observed_estimators) * 6
    if len(results) != expected_cells:
        failures.append(f"expected {expected_cells} graph cells, found {len(results)}")
    counts = Counter(
        (item["dataset"], item["estimator"]) for item in results
    )
    for dataset in observed_datasets:
        for estimator in observed_estimators:
            if counts[(dataset, estimator)] != 6:
                failures.append(
                    f"{dataset}/{estimator} has {counts[(dataset, estimator)]}/6 seasons"
                )
    core_hashes = sorted({item["core_sha256"] for item in results})
    if len(core_hashes) != 1:
        failures.append(f"multiple paper-core hashes: {core_hashes}")
    for item in results:
        expected_dags = {3: 25, 4: 543, 5: 29281}[len(item["variables"])]
        if int(item["dag_count"]) != expected_dags:
            failures.append(
                f"{item['dataset']}/{item['season']}/{item['estimator']} "
                f"has {item['dag_count']} DAGs, expected {expected_dags}"
            )
    paired_hashes: dict[tuple[str, str], set[str]] = {}
    for item in results:
        paired_hashes.setdefault(
            (item["dataset"], item["season"]), set()
        ).add(item["input_sha256"])
    for key, hashes in paired_hashes.items():
        if len(hashes) != 1:
            failures.append(f"input hash mismatch for {key}: {sorted(hashes)}")
    if not diagnostics.empty:
        minimum = float(diagnostics["minimum_loglik_improvement"].min())
        if minimum < -1e-7:
            failures.append(
                f"optimized likelihood degraded below initialization: {minimum}"
            )
    source_root = Path(__file__).resolve().parent
    source_files = (
        source_root / "run_adaptive_ptsem_real_study.py",
        source_root / "analyze_adaptive_ptsem_real_study.py",
        source_root / "make_adaptive_ptsem_recovery_table.py",
        source_root / "plot_adaptive_ptsem_real_study.py",
        source_root / "test_adaptive_ptsem_real_study.py",
    )
    payload = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "graph_cells": len(results),
        "group_counts": {
            f"{dataset}/{estimator}": count
            for (dataset, estimator), count in sorted(counts.items())
        },
        "paper_core_sha256": core_hashes,
        "all_dag_counts_valid": not any("DAGs" in item for item in failures),
        "paired_input_hashes_valid": not any(
            "input hash mismatch" in item for item in failures
        ),
        "minimum_optimization_loglik_improvement": (
            float(diagnostics["minimum_loglik_improvement"].min())
            if not diagnostics.empty
            else None
        ),
        "source_sha256": {
            path.name: sha256(path) for path in source_files if path.exists()
        },
    }
    (root / "final_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    if failures:
        raise RuntimeError("final audit failed: " + "; ".join(failures))


def method_summary_frame(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(["dataset", "unit", "estimator"], as_index=False)
        .agg(
            n_seasons=("season", "nunique"),
            core_chain_recovery_rate=("core_foul_chain", "mean"),
            exact_reference_count=("exact_reference_graph", "sum"),
            exact_reference_rate=("exact_reference_graph", "mean"),
            mean_directed_precision=("directed_precision", "mean"),
            mean_directed_recall=("directed_recall", "mean"),
            mean_directed_f1=("directed_f1", "mean"),
            mean_skeleton_precision=("skeleton_precision", "mean"),
            mean_skeleton_recall=("skeleton_recall", "mean"),
            mean_skeleton_f1=("skeleton_f1", "mean"),
            median_bic_delta_second=("bic_delta_second", "median"),
            minimum_bic_delta_second=("bic_delta_second", "min"),
            total_runtime_sec=("runtime_sec", "sum"),
        )
        .sort_values(["dataset", "estimator"])
    )


def write_report(
    root: Path,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    edges: pd.DataFrame,
    families: pd.DataFrame,
    comparisons: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    lines = [
        "# 最新六族自适应 PT-SEM：NBA 实证结果",
        "",
        "## 分析口径",
        "",
        "- 模型与最终 simulation 使用同一六族外生分布库和同一精确卷积似然。",
        "- `moment` 是论文 simulation 的矩估计 plug-in BIC；`optimization` 是本实证新增的联合约束最大似然 BIC。",
        "- 参考图来自篮球规则，仅用于衡量结构一致性，不是从观测数据中可见的因果真值。",
        "",
        "## 总体结果",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- **{row.dataset} / {row.estimator}**："
            f"核心罚球链恢复率 {row.core_chain_recovery_rate:.1%}；"
            f"参考图精确一致 {int(row.exact_reference_count)}/{int(row.n_seasons)}；"
            f"平均 directed F1={row.mean_directed_f1:.3f}；"
            f"第二优图 BIC 差中位数={row.median_bic_delta_second:.3f}。"
        )
    lines.extend(["", "## 六赛季逐季最优图", ""])
    for row in detail.itertuples(index=False):
        lines.append(
            f"- {row.dataset} / {row.estimator} / {row.season}: "
            f"`{row.edges}`；族=`{row.selected_families}`；"
            f"ΔBIC₂={row.bic_delta_second:.3f}。"
        )
    lines.extend(["", "## 稳定边", ""])
    stable = edges.loc[edges["selection_rate"] >= 5.0 / 6.0]
    if stable.empty:
        lines.append("- 没有达到 5/6 赛季阈值的有向边。")
    else:
        for row in stable.itertuples(index=False):
            marker = "（参考边）" if row.in_reference_graph else "（额外边）"
            lines.append(
                f"- {row.dataset} / {row.estimator}: `{row.edge}` "
                f"{int(row.selected_seasons)}/{int(row.n_seasons)} {marker}"
            )
    lines.extend(["", "## 两种估计方案的差异", ""])
    if comparisons.empty:
        lines.append("- 尚无成对结果。")
    else:
        same = int(comparisons["same_graph"].sum())
        lines.append(
            f"- 完全相同的最优图：{same}/{len(comparisons)} 个数据集–赛季组合。"
        )
        lines.append(
            f"- 矩估计核心链恢复：{comparisons['moment_core_chain'].mean():.1%}；"
            f"optimization 核心链恢复："
            f"{comparisons['optimization_core_chain'].mean():.1%}。"
        )
        lines.append(
            "- 两列 BIC分别是 plug-in BIC 与 MLE-BIC；MLE 分数更低是优化的"
            "定义结果，不能单独作为因果结构更真的证据。"
        )
    lines.extend(["", "## 数值审计", ""])
    if diagnostics.empty:
        lines.append("- 尚无 optimization 局部拟合。")
    else:
        lines.append(
            f"- 共检查 {int(diagnostics['n_local_family_fits'].sum())} 个"
            "局部族拟合；所有有限拟合相对初始化点的最小似然改进为 "
            f"{diagnostics['minimum_loglik_improvement'].min():.3e}。"
        )
        lines.append(
            f"- 最优图节点中有 {int(families['boundary_solution'].sum())}/"
            f"{len(families)} 个族参数位于预设边界阈值附近；这类族标签"
            "应解释为工作分布选择，而不是稳定的物理机制。"
        )
    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- 数据是季度聚合计数，不能恢复同一节内事件的精确时间顺序。",
            "- 同一比赛内各节、同一比赛两队并非严格独立；BIC结构稳定性"
            "不能替代聚类抽样不确定性。",
            "- `FTM <= FTA` 是确定性支持约束，而当前六族局部模型没有显式"
            "编码该不等式；罚球链结果应结合篮球规则解释。",
            "- 分布自由是节点级六族有限库自适应，不是完全非参数分布。",
            "",
            "详细表：`graph_results_detailed.csv`、`method_summary.csv`、"
            "`edge_stability.csv`、`selected_node_families.csv`、"
            "`estimator_comparison.csv`、`optimization_diagnostics.csv`。",
            "",
        ]
    )
    (root / "RESULTS_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def write_paper_ready_report(
    root: Path,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    edges: pd.DataFrame,
    comparisons: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    lines = [
        "# Adaptive-family PT-SEM analysis of NBA play-by-play counts",
        "",
        "## Design",
        "",
        "The analysis used the same node-wise six-family library, exact convolution "
        "likelihood, and exact order-graph dynamic program as the final simulation "
        "suite. Two estimators were evaluated: the simulation's moment plug-in "
        "score and a joint constrained maximum-likelihood score initialized at "
        "the moment estimates. Results were fitted separately in six NBA seasons.",
        "",
        "The three-variable analysis used team-quarter counts of fouls drawn "
        "(FOUL), free-throw attempts (FTA), and free throws made (FTM). The "
        "five-variable analyses added missed field goals (MISS_FG) and rebounds "
        "(REB), at both team-quarter and game-quarter resolutions.",
        "",
        "## Results",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.dataset}, {row.estimator}: the FOUL→FTA→FTM chain was "
            f"recovered in {row.core_chain_recovery_rate:.1%} of seasons; "
            f"the full rule-based reference graph was recovered in "
            f"{int(row.exact_reference_count)}/{int(row.n_seasons)} seasons; "
            f"mean directed F1={row.mean_directed_f1:.3f}."
        )
    lines.extend(["", "The season-specific selected graphs were:", ""])
    for row in detail.itertuples(index=False):
        lines.append(
            f"- {row.dataset}, {row.estimator}, {row.season}: "
            f"`{row.edges}` (second-best ΔBIC={row.bic_delta_second:.3f})."
        )
    stable = edges.loc[edges["selection_rate"] >= 5.0 / 6.0]
    lines.extend(["", "## Cross-season stability", ""])
    for row in stable.itertuples(index=False):
        role = "reference" if row.in_reference_graph else "additional"
        lines.append(
            f"- {row.dataset}, {row.estimator}: `{row.edge}` appeared in "
            f"{int(row.selected_seasons)}/{int(row.n_seasons)} seasons "
            f"({role} edge)."
        )
    lines.extend(["", "## Numerical audit and interpretation", ""])
    if not diagnostics.empty:
        lines.append(
            f"The audit covered {int(diagnostics['n_local_family_fits'].sum())} "
            "node–parent-set–family fits. The smallest finite log-likelihood "
            "improvement relative to an initialization point was "
            f"{diagnostics['minimum_loglik_improvement'].min():.3e}; hence no "
            "optimized local score was worse than its retained initialization."
        )
    if not comparisons.empty:
        lines.append(
            f"The two estimators selected exactly the same graph in "
            f"{int(comparisons['same_graph'].sum())}/{len(comparisons)} "
            "matched dataset-season analyses. MLE-BIC is necessarily no larger "
            "than the corresponding plug-in score and this numerical improvement "
            "must not itself be interpreted as causal validation."
        )
    lines.extend(
        [
            "",
            "The rule-based graph is an external structural reference rather than "
            "an observed causal ground truth. Quarter aggregation removes within-"
            "quarter temporal order, observations from the same game are dependent, "
            "and the current local family library does not explicitly encode the "
            "deterministic FTM≤FTA support restriction. The selected exogenous "
            "families should therefore be interpreted as working distributions, "
            "particularly when their parameters lie near a family boundary.",
            "",
        ]
    )
    (root / "PAPER_READY_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/adaptive_ptsem_real_study"),
    )
    args = parser.parse_args()
    root = args.results_dir.resolve()
    results = load_graph_results(root)
    detail = graph_detail_frame(results)
    summary = method_summary_frame(detail)
    edges = edge_stability_frame(results)
    families = family_selection_frame(results)
    comparisons = estimator_comparison_frame(detail)
    diagnostics = optimization_diagnostics(root)
    write_final_audit(root, results, diagnostics)
    atomic_write_csv(detail, root / "graph_results_detailed.csv")
    atomic_write_csv(summary, root / "method_summary.csv")
    atomic_write_csv(edges, root / "edge_stability.csv")
    atomic_write_csv(families, root / "selected_node_families.csv")
    atomic_write_csv(comparisons, root / "estimator_comparison.csv")
    atomic_write_csv(diagnostics, root / "optimization_diagnostics.csv")
    write_report(
        root,
        detail,
        summary,
        edges,
        families,
        comparisons,
        diagnostics,
    )
    write_paper_ready_report(
        root,
        detail,
        summary,
        edges,
        comparisons,
        diagnostics,
    )
    print(f"Analysis written to {root}", flush=True)


if __name__ == "__main__":
    main()
