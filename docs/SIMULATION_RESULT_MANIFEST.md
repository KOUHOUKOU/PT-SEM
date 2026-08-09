# Simulation result manifest

| Sweep | Raw file | Paper use |
|---|---|---|
| dimension | `data/simulation/raw/experiment1_d_sweep_raw.csv` | Figures 1–3 |
| sample size | `data/simulation/raw/experiment2_N_sweep_raw.csv` | Figures 1–3 |
| average indegree | `data/simulation/raw/experiment3_kin_sweep_raw.csv` | Figures 1–3 |

The corresponding summary, family-assignment audit, family-confusion, and family-metric files are under `data/simulation/summaries/`. `final_suite_manifest.json` is the formal run record, and `paper_final_all_sweeps_summary.csv` is the consolidated reporting table.

All byte sizes and SHA-256 values are machine-readable in `manifests/SHA256SUMS.csv`. Paper PDFs are separately frozen in `figures/manuscript/` so regenerated PDF metadata cannot obscure whether the exact submitted artifact was retained.
