"""Render the final four- and five-node expanded-variable recovery tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from make_adaptive_ptsem_recovery_table import render_table


ROOT = Path("outputs/expanded_candidate_study")


def render(stem: str, caption: str) -> None:
    frame = pd.read_csv(ROOT / f"{stem}.csv")
    parts = frame["exact_recovery"].str.split("/", expand=True)
    frame["exact_recovery_count"] = parts[0].astype(int)
    frame["n_seasons"] = parts[1].astype(int)
    render_table(frame, ROOT, caption)
    for suffix in ("png", "pdf"):
        (ROOT / f"table_graph_recovery.{suffix}").replace(
            ROOT / f"{stem}.{suffix}"
        )


def main() -> None:
    render(
        "final_four_node_table",
        "Team-quarter recovery across six NBA seasons for "
        "FOUL→FTA→FTM and FOUL→PERS_FOUL_DRAWN. PB-SCM rows are forced "
        "author-code applications outside the full unit-thinning domain.",
    )
    render(
        "final_five_node_table",
        "Team-quarter recovery across six NBA seasons for "
        "FOUL→FTA→FTM with personal-foul and loose-ball-foul leaves. "
        "PB-SCM rows are forced author-code applications outside the full "
        "unit-thinning domain.",
    )


if __name__ == "__main__":
    main()
