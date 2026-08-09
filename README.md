# PT-SEM: causal DAG identification for count data

Code, data and results for *Causal DAG Identification for Count Data via
Poisson-Thinning Structural Equation Models*.

Manuscript of record: `PT_SEM_jmlr (51).pdf`, SHA-256 `7f485b4e…`.

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

## Check it in one command

No scientific environment needed; `pandas` and `pypdf` suffice.

```bash
python scripts/build_paper_objects.py
```

It compares every object with what the manuscript actually contains and prints
one line per object:

```
Figure 1   figure1_all_poisson.pdf   MATCH   514a0d4273dd807d
...
ALL PAPER OBJECTS REPRODUCED
```

Integrity of the shipped artifacts:

```bash
python scripts/verify_frozen_results.py
python -m unittest discover -s tests
```

## Reproducing the results

Two pinned environments are required and must not be merged; they differ only
in NumPy. See `docs/ENVIRONMENT.md`.

| Line | Environment | Guide |
|---|---|---|
| Figures 1–4 | `requirements-simulation.txt` | `docs/SIMULATION_REPRODUCTION.md` |
| Figures 5–6, Table 3 | `requirements-nba.txt` | `docs/NBA_REPRODUCTION.md` |

`docs/CODE_MAP.md` states what every program reads and writes and how they
chain.

Redrawing the figures from the committed results takes seconds. Refitting from
zero took 3 h 13 min and 3 h 38 min for the two simulation suites and about
2.2 h for the ten NBA seasons, on 24 logical cores with 12 workers.

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
