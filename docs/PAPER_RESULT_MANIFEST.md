# Paper result manifest

Authoritative manuscript: `PT_SEM_jmlr (51).pdf`, 32 pages, SHA-256
`7f485b4ebc23a2db8db836776a0d87215e42d8ba1fc6bc768ba61cd488a8467f`.

## Figure numbering

The manuscript was reorganised so that the all-Poisson experiment is presented
first. The programs that draw the figures keep their original output names, so
**the file names do not match the manuscript's figure numbers.** This table is
the mapping; `manifests/PAPER_OBJECTS.csv` is its machine-readable form, and
`outputs/` delivers every object under its manuscript name.

| Manuscript | Page | Delivered as | Drawn by | Produced file name |
|---|---:|---|---|---|
| Figure 1 | 15 | `outputs/figures/figure1_all_poisson.pdf` | `scripts/plot_figure4.py` | `fig4_all_poisson_library_cost_final.pdf` |
| Figure 2 | 16 | `outputs/figures/figure2_dag_recovery_f1.pdf` | `scripts/plot_mixed_family.py` | `fig1_directed_f1_final.pdf` |
| Figure 3 | 17 | `outputs/figures/figure3_coefficient_mape.pdf` | `scripts/plot_mixed_family.py` | `fig2_conditional_alpha_mape_final.pdf` |
| Figure 4 | 17 | `outputs/figures/figure4_family_selection.pdf` | `scripts/plot_mixed_family.py` | `fig3_working_family_diagnostics_final.pdf` |
| Figure 5 | 18 | `outputs/figures/figure5_nba_reference_dag.pdf` | `experiments/nba/scripts/make_nba_final_figures.py` | `fig5_rule_implied_nba_graph_final.pdf` |
| Figure 6 | 19 | `outputs/figures/figure6_nba_season_estimates.pdf` | `experiments/nba/scripts/make_nba_final_figures.py` | `fig6_nba_diagnostics_final.pdf` |
| Table 3 | 18 | `outputs/tables/table3_nba_structural_recovery.csv` | `experiments/nba/scripts/summarize_nba_extension.py` | `table2_nba_graph_recovery.csv` |

Tables 1 and 2 are design settings stated in the manuscript text, not computed
artifacts. `scripts/verify_frozen_results.py` checks that the executed sweep
grid matches Table 2.

## What each object is verified against

`scripts/build_paper_objects.py` compares every object with what the manuscript
actually contains. Figures are compared by the PDF content stream, which LaTeX
preserves when a figure is included, so a match proves the shipped artifact is
the one printed in the paper. The result is written to
`outputs/REPRODUCTION_REPORT.md`.

## Numerical cores

Two cores are shipped because the two experiment lines were computed with
different ones, and both are recorded inside the frozen results:

| Core | SHA-256 | Used by |
|---|---|---|
| `src/d.py` | `70da3d96…` | Figures 1–4 |
| `experiments/nba/core/d.py` | `d14c13f5…` | Figures 5–6, Table 3 |

`src/d.py` is the corrected `ptsem_final_nb_exact_v2` build. The NBA core is the
historical build named in every `data/nba/graph_json/*.json`; see
`docs/UNRESOLVED_PROVENANCE.md` for why it is retained and what was measured
about it.

## Text claims backed by committed results

| Claim | Backed by |
|---|---|
| average directed-edge F1 of 0.975 | `results/nba/nba_main_graph_recovery_summary.csv` |
| reference DAG recovered in nine of the ten seasons | `results/nba/nba_main_graph_recovery_by_season.csv` |
| FOUL→FTA coefficient ranges 1.187–1.297, mean 1.238 | `results/nba/tables/table3_nba_coefficient_summary.csv` |
