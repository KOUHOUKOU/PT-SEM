# PT-SEM: causal DAG identification for count data

Code, data and results for *Causal DAG Identification for Count Data via
Poisson-Thinning Structural Equation Models*.

Manuscript of record: `PT_SEM_jmlr (51).pdf`, SHA-256 `7f485b4e…`.

Repository: <https://github.com/KOUHOUKOU/PT-SEM>

## What this repository delivers

`outputs/` contains the manuscript's objects and nothing else:

```
outputs/figures/figure1_all_poisson.pdf
outputs/figures/figure2_dag_recovery_f1.pdf
outputs/figures/figure3_coefficient_mape.pdf
outputs/figures/figure4_family_selection.pdf
outputs/figures/figure5_nba_reference_dag.pdf
outputs/figures/figure6_nba_season_estimates.pdf
outputs/tables/table3_nba_structural_recovery.csv
outputs/REPRODUCTION_REPORT.md
```

The manuscript renumbered its figures, so the file names the plotting programs
produce no longer match the figure numbers a reader sees.
`docs/PAPER_RESULT_MANIFEST.md` is that mapping, and `outputs/` already applies
it.

## Three ways to use this, in increasing depth

### 1. Confirm the shipped objects are the paper's (30 seconds)

```bash
git clone https://github.com/KOUHOUKOU/PT-SEM.git
cd PT-SEM
pip install -r requirements-verify.txt
python scripts/build_paper_objects.py
```

Two packages, no scientific environment. It compares every delivered object
with the content stream embedded in the manuscript and prints one line each:

```
Figure 1   figure1_all_poisson.pdf   MATCH   514a0d4273dd807d
...
ALL PAPER OBJECTS REPRODUCED
```

A `MATCH` means the file in `outputs/` is byte-identical to what the paper
prints. The same run writes `outputs/REPRODUCTION_REPORT.md`. Integrity of the
committed artifacts:

```bash
python scripts/verify_frozen_results.py
python -m unittest discover -s tests
```

### 2. Redraw the figures from the committed results (minutes)

This regenerates Figures 1-4 rather than checking a shipped copy, so it
demonstrates that the committed data really produces the published figures.

```bash
python -m venv .venv-simulation
.venv-simulation/Scripts/python -m pip install -r requirements-simulation.txt
.venv-simulation/Scripts/python scripts/plot_figure4.py
.venv-simulation/Scripts/python scripts/plot_mixed_family.py
python scripts/build_paper_objects.py
```

The last command now reports `final_figures/...` in its `produced from`
column, meaning it verified what was just drawn. Figures 5-6 are not redrawn
here because that needs the NBA refit below.

### 3. Refit everything from the seeds (hours)

Nothing is downloaded; all simulated data is generated from master seed
`20260622`. See `docs/SIMULATION_REPRODUCTION.md` and
`docs/NBA_REPRODUCTION.md` for the commands, and `docs/CODE_MAP.md` for what
each program reads and writes.

Recorded runtimes on 24 logical cores with 12 workers: 3 h 13 min for the
all-Poisson suite, 3 h 38 min for the mixed-family suite, about 2.2 h for the
ten NBA seasons.

## Environments

Three requirement files, because the two experiment lines were computed under
different NumPy versions and must not share one environment:

| File | Covers | NumPy |
|---|---|---|
| `requirements-verify.txt` | integrity checks only | not used |
| `requirements-simulation.txt` | Figures 1–4 | 2.2.6 |
| `requirements-nba.txt` | Figures 5–6, Table 3 | 1.26.4 |

`scripts/check_environment.py simulation` and `... nba` verify the interpreter
and exit non-zero on a mismatch; the refit entry points refuse to run in the
wrong one. See `docs/ENVIRONMENT.md` for why the split is enforced.

## Layout

```
src/                  simulation core and frameworks (Figures 1-4)
config/               locked simulation designs
scripts/              run / summarize / plot / verify entry points
experiments/nba/      NBA study: historical scripts and their core
results/              committed results for all three lines
data/nba/             processed NBA inputs and fitted graphs
outputs/              the manuscript's objects
manifests/            SHA-256 inventories
docs/, provenance/, audit/   reproduction guides and provenance
```

Two numerical cores are shipped on purpose: `src/d.py` for Figures 1–4 and
`experiments/nba/core/d.py` for Figures 5–6 and Table 3, matching what is
recorded inside the frozen results. `docs/UNRESOLVED_PROVENANCE.md` explains
why, and what was measured about the difference.

## Data

Simulated data is generated from master seed `20260622`; nothing is downloaded.
The NBA study uses a Kaggle play-by-play dataset that is not redistributable —
the processed team-quarter derivatives the study consumes are committed and
hash-frozen. See `provenance/DATASET_LOCATION.md`.

PB-SCM and PB-SCM-PGF are fetched, not redistributed; see `docs/THIRD_PARTY.md`.

## License and citation

MIT, see `LICENSE`. Citation metadata in `CITATION.cff`. Third-party components
keep their own licenses.
