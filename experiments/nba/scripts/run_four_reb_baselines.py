"""Run the shared baseline suite for FOUL, FTA, FTM, and REB."""

from __future__ import annotations

import run_four_var_baseline_comparison as baseline


baseline.VARIABLES = ["FOUL", "FTA", "FTM", "REB"]
baseline.REFERENCE_EDGES = {
    ("FOUL", "FTA"),
    ("FTA", "FTM"),
    ("FTA", "REB"),
}


if __name__ == "__main__":
    baseline.main()
