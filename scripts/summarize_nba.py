#!/usr/bin/env python3
"""Create the NBA structural-recovery table from per-season results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEASONS = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2015, 2025))


def exogenous_mean(fit: dict) -> float:
    """Return the mean implied by a fitted exogenous-family record."""
    family, parameters = fit["family"], fit["params"]
    if family == "Poisson":
        return float(parameters["lam"])
    if family == "NB":
        return float(parameters["r"] * (1.0 - parameters["p"]) / parameters["p"])
    if family == "ZIP":
        return float((1.0 - parameters["rho"]) * parameters["lam"])
    if family == "Geom":
        return float((1.0 - parameters["p"]) / parameters["p"])
    if family in {"Binomial", "Bernoulli"}:
        return float(parameters.get("n", 1) * parameters["p"])
    raise ValueError(family)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ten ordered season records for each recovery method."""
    metric_names = (
        "skeleton_precision",
        "skeleton_recall",
        "skeleton_f1",
        "directed_precision",
        "directed_recall",
        "directed_f1",
    )
    rows = []
    for method, group in detail.groupby("method", sort=False):
        if tuple(group["season"]) != SEASONS:
            raise RuntimeError(f"{method} does not contain the ten seasons in order")
        row: dict[str, object] = {
            "method": method,
            "n_seasons": len(group),
            "exact_recovery_count": int(group["exact_recovery"].sum()),
            "estimator": str(group["estimator"].iloc[0]),
        }
        for metric_name in metric_names:
            row[f"mean_{metric_name}"] = float(group[metric_name].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "nba" / "summaries" / "nba_all_methods_by_season.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "tables" / "nba_structural_recovery.csv",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summarize(pd.read_csv(args.input.resolve())).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
