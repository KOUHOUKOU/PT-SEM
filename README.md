# PT-SEM reproducibility repository

This repository freezes the programs, inputs, result tables, and paper figures for **“Causal DAG Identification for Count Data via Poisson-Thinning Structural Equation Models”** by Penggang Gao, Ming Cai, and Hisayuki Hara.

The repository is result-complete: the paper figures can be regenerated from committed result tables without rerunning the expensive estimators. It also preserves the historical simulation and NBA programs needed for a fresh run. The authoritative manuscript checked during this freeze was the 33-page `PT_SEM_jmlr (44).pdf` (SHA-256 `70b2cf02b13435810b65c4d428caa774b210e539b447215f84f6cd0ada9996fa`). The manuscript itself is not redistributed here.

## Important scientific notice

Figures 1–4 were produced with the historical frozen scoring implementation. A later audit found that its negative-binomial thinning convolution used an invalid `hyp1f1` identity on part of the parameter domain. The exact paper artifacts are retained for reproducibility and are **not silently replaced** by corrected post-paper calculations. See [docs/UNRESOLVED_PROVENANCE.md](docs/UNRESOLVED_PROVENANCE.md) and [audit/corrected_nb_convolution](audit/corrected_nb_convolution). The NBA paper artifacts are unaffected by any repository repackaging.

## Quick start

Python 3.11.9 is the recorded environment. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_frozen_results.py
python scripts/generate_figures.py
python -m unittest discover -s tests -v
```

Generated PDFs are written to `figures/generated/`. The six immutable paper PDFs are in `figures/manuscript/`.

## Paper-to-repository map

| Paper item | Frozen inputs | Regeneration entry point | Status |
|---|---|---|---|
| Figure 1 | mixed-family raw results and saved plot data | `scripts/generate_figures.py` | exact paper PDF preserved; redraw verified from saved data |
| Figure 2 | mixed-family raw results and saved plot data | `scripts/generate_figures.py` | exact paper PDF preserved; redraw verified from saved data |
| Figure 3 | family-selection summaries/confusion tables | `scripts/generate_figures.py` | exact paper PDF preserved; redraw verified from saved data |
| Figure 4 | 24,000 all-Poisson method rows and validated summary | `scripts/generate_figures.py` | exact paper PDF preserved; formal validation passed |
| Figure 5 | reference NBA graph | `scripts/generate_figures.py` | exact paper PDF preserved |
| Figure 6 | ten-season coefficients and selected working families | `scripts/generate_figures.py` | exact paper PDF preserved |
| Table 3 | `results/nba/tables/table2_nba_graph_recovery.csv` | `scripts/verify_frozen_results.py` | values match manuscript |

## Repository layout

- `experiments/simulation/legacy`: immutable historical six-family implementation and runner.
- `experiments/simulation/all_poisson_snapshot`: exact all-Poisson framework, configuration, and frozen scoring module.
- `experiments/nba/scripts`: aggregation, fitting, baseline, summarization, and figure programs.
- `data`: committed simulation results, all-Poisson results, and processed ten-season NBA team-quarter data.
- `results/nba`: fitted graphs, coefficients, family labels, and manuscript tables.
- `figures/manuscript`: the six PDFs embedded in the manuscript.
- `manifests`: cryptographic inventory of the frozen publication artifacts.
- `docs`: audit trail, settings, data provenance, runtimes, and result classifications.

## Fresh computation

The mixed-family formal suite uses `R=100`, master seed `20260622`, both coefficient regimes, and all three sweeps. It requires the pinned third-party baselines; run `experiments/simulation/legacy/setup_external_baselines.py` before the suite. A full run is computationally expensive and is not part of the quick verification path.

The NBA raw play-by-play files are not committed because they total about 1 GB. Download Kaggle dataset `shufinskiy/nba-play-by-play-data-2015-to-2025`, version 8 (released 2025-06-26), then follow [docs/NBA_REPRODUCTION.md](docs/NBA_REPRODUCTION.md). The committed processed files allow the estimator and all paper summaries to be rerun without redistributing those raw files.

## Third-party code and licensing

PB-SCM and PB-SCM-PGF are fetched at pinned commits and are excluded from this MIT-licensed repository because their upstream repositories did not provide a license at freeze time. See [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md). Data licensing is documented separately in `provenance/`.

## Integrity

Run `python scripts/verify_frozen_results.py` at any time. It verifies SHA-256 hashes, simulation completeness, all-Poisson validation, NBA row/season constraints, and the manuscript Table 3 values.
