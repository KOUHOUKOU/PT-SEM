# Reproducibility report

Freeze audit date: 2026-08-05.

## Outcome

The publication repository is complete for the current manuscript: source programs, configurations, seeds, committed result-level data, ten-season processed NBA data, result tables, exact paper PDFs, environment records, licensing notes, and SHA-256 manifests are present.

## Verification performed

- 124 immutable publication files passed byte-size and SHA-256 verification.
- The mixed-family suite manifest is complete at R=100 with master seed 20260622; all three consolidated raw sweeps loaded and passed range/completeness checks.
- The all-Poisson result has 24,000 successful method rows, 4,000 paired datasets, 240 complete method-setting cells, and a passed formal validation report.
- The ten NBA processed files contain exactly 95,808 team-quarter rows, seasons 2015-16 through 2024-25, periods 1–4 only, nonnegative counts, and no `FTM > FTA` rows.
- The proposed NBA graph has exact skeleton recovery in 10/10 seasons and exact directed recovery in 9/10; 2022-23 is the sole direction mismatch.
- All six values for each of the five methods in manuscript Table 3 match the committed result table to manuscript rounding.
- All six figures regenerated successfully under Python 3.11.9.
- At 200 DPI, regenerated Figures 1, 2, 4, 5, and 6 are pixel-exact against the manuscript PDFs after rasterization. Figure 3 is data-exact and visually equivalent but not pixel-exact because the final manuscript-only label/layout polish was not preserved as a standalone source file. The exact Figure 3 PDF is committed and hash-frozen.
- Four automated unit tests passed.

## Expensive runs not repeated during packaging

The full R=100 mixed-family Monte Carlo suite, the full all-Poisson suite, and all ten NBA optimizations were not recomputed from zero during this packaging pass. Their complete frozen outputs and strict validation reports were verified instead. Fresh-run commands and expected runtimes are documented in the reproduction guides.

## Status by item

| Item | Artifact | Numerical inputs | Fresh full rerun |
|---|---|---|---|
| Figures 1–2 | VERIFIED EXACT | VERIFIED EXACT | not repeated |
| Figure 3 | exact PDF preserved; redraw VISUALLY EQUIVALENT | VERIFIED EXACT | not repeated |
| Figure 4 | VERIFIED EXACT | VERIFIED EXACT | not repeated |
| Figures 5–6 | VERIFIED EXACT | VERIFIED EXACT | not repeated |
| Table 3 | VERIFIED EXACT | VERIFIED EXACT | not repeated |

The N=2400 manuscript-table discrepancy, absence of historical Git metadata, missing historical root manifest, unlicensed external baselines, and post-paper NB-convolution defect remain explicitly disclosed in `UNRESOLVED_PROVENANCE.md`.
