# JMLR final reproducibility report

## Protection and source state

- Project root: `the recovered source tree`.
- The project root and its parents contain no Git repository metadata. Commit,
  branch, status, and diff are therefore unavailable and are recorded as such
  in `SOURCE_STATE.txt`; `git_diff.patch` explicitly states N/A.
- No commit, push, reset, deletion, move, or rename was performed.
- No original experiment source or original result file was modified.
- Current paper PDFs were not overwritten during provenance discovery,
  saved-result reproduction, experiment execution, or preliminary plotting.

## Candidate programs found

The full `.py`, shell/config, result-file, export-command, and filename search
found these meaningful plotting candidates:

1. `build_final_paper_figures.py` - accepted formal source. It is the only
   program that writes every exact Figure 1-6 final filename, contains both the
   original raw-result aggregation and saved-plotdata redraw chain, matches the
   current method labels/panel layouts, and reproduces current pixels.
2. `plot_final_main_text_figures.py` - rejected: older output stems and a
   different Figure 2 metric/name.
3. `polish_final_main_text_figures.py` - rejected: polishes the older stems,
   not the six target files.
4. `plot_revised_section5.py` - rejected: historical section package with
   different numbering, layouts, and figure claims.
5. `make_main_combined_figures.py` - rejected: directed-F1/runtime combined
   figures, not the target files.
6. `build_complete_paper_results_package.py` - rejected: numbered package
   figures, including an older 2x2 All-Poisson layout.
7. `build_experiment4_figure_summary.py` - rejected: contact-sheet/summary
   utility for an earlier experiment.
8. `plot_directed_f1_sd_preview.py` - rejected: SD preview rather than the
   stated Monte Carlo CI figure.
9. `ptsem_extended_experiments.py` plot functions - accepted as upstream raw
   All-Poisson experiment provenance, rejected as the final Figure 4 plotter.

No `.ipynb` files exist in the searched project. No PDF-merging chain produces
the six final figures. The legacy `MAIN_TEXT_PDFS_FOR_UPLOAD/` and
`ALL_PAPER_FIGURE_PDFS_FOR_UPLOAD/` copies have older text/layout and were not
mistaken for the current `final_paper_figures/` artifacts.

## Figure 1-6 provenance and saved-result reproduction

The complete provenance is documented in the README table and frozen
manifests. In summary:

- Figures 1-3: `ptsem_six_family_final_suite.py` and `d.py` -> three formal
  per-replication raw CSVs -> `build_final_paper_figures.py` -> saved plotdata
  -> exact final PDFs.
- Figure 4 (pre-extension): `ptsem_extended_experiments.py` All-Poisson
  restricted N-sweep -> `synthetic_per_replication.csv` -> formal plotter.
- Figure 5: deterministic four-edge rule-implied graph in the formal plotter.
- Figure 6: frozen Basketball optimization graph JSONs and coefficient CSV,
  audited selected-family CSV from `run_nba_audit`, then the formal plotter.

The full formal plotting program was rerun with output redirected under the
snapshot. Its eight core plotdata CSVs are byte-for-byte identical to the
current files. The saved-plotdata redraw path then reproduced Figures 1-6.
At fixed 200 DPI, all six reproduced PNG SHA256 values are exactly identical
to the current paper renders. PDF hashes differ only because PDF metadata is
regenerated. See `reproduced_figures/verification_report.json`.

## Existing All-Poisson sample-size truth

The current saved result is restricted/common only, not extended or mixed.
It has all-Poisson exogenous noise with lambda independently Uniform(2,10),
`d=8`, average indegree 1.5, exact 12-edge DAGs, `max_parents=5`, coefficients
Uniform(0.15,0.85), the required eight N values, and `R=100`. All seven
restricted methods have 100 successful replications per N, share identical
truth within replication, and have no duplicate seed or NaN F1. Its mean/SD/
SE/CI values reproduce the old Figure 4 plotdata to numerical roundoff without
reading the PDF.

## New All-Poisson experiment status

Final counts, failure totals, per-cell R checks, output hashes, and replacement
status are written here after the resumable formal run and final visual QA.

## Reproduction command

Portable shell entry point:

```bash
bash scripts/run_reproduction.sh --all
```

Tested PowerShell-equivalent commands on the snapshot host:

```powershell
python scripts/verify_existing_figures.py
python scripts/run_all_poisson_missing_sweeps.py --workers 12
python scripts/summarize_all_poisson.py
python scripts/plot_all_poisson_final.py
```

The snapshot-wide hash manifest is `SHA256SUMS.txt`; detailed original/snapshot
path mapping is `MANIFEST.csv`.
