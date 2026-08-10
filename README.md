# PT-SEM: causal DAG identification for count data

Code, data and results for *Causal DAG Identification for Count Data via
Poisson-Thinning Structural Equation Models*.

Repository: <https://github.com/KOUHOUKOU/PT-SEM>

## What is provided

`outputs/` holds the paper's figures and table under their paper names:

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

`docs/PAPER_RESULT_MANIFEST.md` maps each of these to the program and data that
produce it.

## Verification

```bash
git clone https://github.com/KOUHOUKOU/PT-SEM.git
cd PT-SEM
pip install -r requirements-verify.txt
python scripts/build_paper_objects.py
python scripts/verify_frozen_results.py
python -m unittest discover -s tests
```

`build_paper_objects.py` compares each delivered object with its expected
digest and writes `outputs/REPRODUCTION_REPORT.md`. `verify_frozen_results.py`
checks the committed artifacts against `manifests/SHA256SUMS.csv` and the
structural invariants of each experiment line. Neither needs a scientific
environment.

Optionally, the released paper objects can be cross-checked against a supplied
manuscript PDF:

```bash
python scripts/build_paper_objects.py --manuscript <path>
```

## Reproduction

Figures 1–4 can be regenerated from the committed simulation results:

```bash
python -m venv .venv-simulation
.venv-simulation/Scripts/python -m pip install -r requirements-simulation.txt
.venv-simulation/Scripts/python scripts/plot_figure4.py
.venv-simulation/Scripts/python scripts/plot_mixed_family.py
python scripts/build_paper_objects.py
```

Refitting from zero requires no downloaded data: all simulated datasets are
generated from master seed `20260622`. Commands are in
`docs/SIMULATION_REPRODUCTION.md` and `docs/NBA_REPRODUCTION.md`;
`docs/CODE_MAP.md` states what each program reads and writes.

Recorded runtimes on 24 logical cores with 12 workers: 3 h 13 min for the
all-Poisson suite, 3 h 38 min for the mixed-family suite, about 2.2 h for the
ten NBA seasons.

## Environments

The two experiment lines were computed under different NumPy versions and
require separate environments.

| File | Covers | NumPy |
|---|---|---|
| `requirements-verify.txt` | verification and tests | — |
| `requirements-simulation.txt` | Figures 1–4 | 2.2.6 |
| `requirements-nba.txt` | Figures 5–6, Table 3 | 1.26.4 |

`scripts/check_environment.py simulation` and `... nba` check the interpreter
and exit non-zero on a mismatch; the refit entry points refuse to run in the
wrong one. See `docs/ENVIRONMENT.md`.

## Layout

```
src/                  simulation core and frameworks (Figures 1-4)
config/               locked simulation designs
scripts/              run / summarize / plot / verify entry points
experiments/nba/      NBA study scripts and their numerical core
results/              committed results for all three experiment lines
data/nba/             processed NBA inputs and fitted graphs
outputs/              the paper's figures and table
manifests/            SHA-256 inventories and the paper-object mapping
docs/, provenance/, audit/   reproduction guides and provenance
```

Two numerical cores are shipped: `src/d.py` for Figures 1–4 and
`experiments/nba/core/d.py` for Figures 5–6 and Table 3, matching the core
recorded in each line's frozen results. See `docs/UNRESOLVED_PROVENANCE.md`.

## Data

Simulated data is generated from master seed `20260622`. The NBA study uses a
Kaggle play-by-play dataset that is not redistributed here; the processed
team-quarter counts consumed by the analysis are committed and hash-frozen. See
`provenance/DATASET_LOCATION.md`.

PB-SCM and PB-SCM-PGF are fetched rather than redistributed; see
`docs/THIRD_PARTY.md`.

## License and citation

MIT, see `LICENSE`. Citation metadata in `CITATION.cff`. Third-party components
keep their own licenses.
