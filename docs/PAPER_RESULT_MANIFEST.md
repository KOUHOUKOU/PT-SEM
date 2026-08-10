# Paper result manifest

Mapping between the paper's objects and the artifacts and programs in this
repository. `manifests/PAPER_OBJECTS.csv` is the machine-readable form;
`outputs/` delivers each object under its paper name.

| Paper object | Subject | Delivered as | Drawn by | Produced file |
|---|---|---|---|---|
| Figure 1 | all-Poisson setting | `outputs/figures/figure1_all_poisson.pdf` | `scripts/plot_figure4.py` | `fig4_all_poisson_library_cost_final.pdf` |
| Figure 2 | F1 for DAG recovery | `outputs/figures/figure2_dag_recovery_f1.pdf` | `scripts/plot_mixed_family.py` | `fig1_directed_f1_final.pdf` |
| Figure 3 | thinning-coefficient MAPE | `outputs/figures/figure3_coefficient_mape.pdf` | `scripts/plot_mixed_family.py` | `fig2_conditional_alpha_mape_final.pdf` |
| Figure 4 | exogenous-family selection | `outputs/figures/figure4_family_selection.pdf` | `scripts/plot_mixed_family.py` | `fig3_working_family_diagnostics_final.pdf` |
| Figure 5 | NBA reference DAG | `outputs/figures/figure5_nba_reference_dag.pdf` | `experiments/nba/scripts/make_nba_final_figures.py` | `fig5_rule_implied_nba_graph_final.pdf` |
| Figure 6 | season-wise NBA estimates | `outputs/figures/figure6_nba_season_estimates.pdf` | `experiments/nba/scripts/make_nba_final_figures.py` | `fig6_nba_diagnostics_final.pdf` |
| Table 3 | NBA structural recovery | `outputs/tables/table3_nba_structural_recovery.csv` | `experiments/nba/scripts/summarize_nba_extension.py` | `table2_nba_graph_recovery.csv` |

Program output names differ from the paper's numbering; the mapping above is
authoritative.

Tables 1 and 2 state design settings rather than computed artifacts.
`scripts/verify_frozen_results.py` checks that the executed sweep grid matches
Table 2.

## Verification

`scripts/build_paper_objects.py` compares each object with the digest recorded
in `manifests/PAPER_OBJECTS.csv`: figures by PDF content stream, the table by
file digest. The result is written to `outputs/REPRODUCTION_REPORT.md`.
Passing `--manuscript <path>` additionally checks that each figure stream is
embedded in the supplied PDF.

## Numerical cores

| Core | SHA-256 | Used by |
|---|---|---|
| `src/d.py` | `70da3d96…` | Figures 1–4 |
| `experiments/nba/core/d.py` | `d14c13f5…` | Figures 5–6, Table 3 |

`src/d.py` is the `ptsem_final_nb_exact_v2` build. The NBA core is the build
recorded in every `data/nba/graph_json/*.json`. See
`docs/UNRESOLVED_PROVENANCE.md`.

## Text values backed by committed results

| Value | Source |
|---|---|
| average directed-edge F1 of 0.975 | `results/nba/nba_main_graph_recovery_summary.csv` |
| reference DAG recovered in nine of the ten seasons | `results/nba/nba_main_graph_recovery_by_season.csv` |
| FOUL→FTA coefficient range 1.187–1.297, mean 1.238 | `results/nba/tables/table3_nba_coefficient_summary.csv` |
