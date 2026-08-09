# NBA reproduction

## Pinned environment is required

Refitting the NBA results is only bit-reproducible under the versions pinned in
`requirements-nba.txt`. This is not a formality. A 2015-16 control run reproduced
the committed result exactly under the pinned NumPy 1.26.4 (score delta 0,
all five node BICs delta 0), while the same code under NumPy 2.2.6 gave a
score delta of 2.3e-05 and flipped one node's selected exogenous family.

```bash
python -m venv .venv-nba
.venv-nba/Scripts/python -m pip install -r requirements-nba.txt
.venv-nba/Scripts/python scripts/check_environment.py nba
```

Use `.venv-nba/bin/python` instead of `.venv-nba/Scripts/python` on Linux and macOS.
`scripts/check_environment.py` exits non-zero on any mismatch, and the refit
entry point refuses to run in an unpinned interpreter.

Hash-only verification is environment-independent and does not need the venv.

## Result-only verification and figures

```bash
python scripts/verify_frozen_results.py --section nba
python scripts/build_paper_objects.py
```

The first checks the committed processed data and result tables; the second
verifies Figures 5-6 and Table 3 against the manuscript and refreshes
`outputs/`. Neither needs the pinned environment.

## Refitting from the committed processed data

```bash
.venv-nba/Scripts/python scripts/reproduce_nba.py
```

This driver produces exactly three deliverables under
`scratch/nba_reproduction/deliverables`: the recovered graph structure per
season, the proposed-method row of manuscript Table 3, and Figures 5 and 6.
Everything else the historical scripts emit (per-family local BIC tables,
optimizer diagnostics, run manifests, staged inputs) stays under
`scratch/nba_reproduction/intermediate`. The driver prints a per-season
structure comparison against the committed results and exits non-zero on any
mismatch. Run `python scripts/build_paper_objects.py` afterwards to refresh
`outputs/` and re-verify against the manuscript.

Useful flags: `--seasons 2015-16 ...` to refit a subset, `--workers N`, and
`--work-dir DIR`. Figures 5-6 and the Table 3 comparison require all ten
seasons, so a subset run reports the per-season structure only.

The original fits took roughly 10-16 minutes per season, about 2.2 hours for
all ten. Exact runtime depends on CPU, optimizer, and BLAS.

The four baseline methods in Table 3 are not refitted by this driver. PB-SCM
and PB-SCM-PGF are not redistributed here (see `docs/THIRD_PARTY.md`), so their
rows and the ODS and PC-RCIT rows must be reproduced from
`experiments/nba/scripts/run_expanded_candidate_baselines.py` with the upstream
sources obtained separately.

## From raw Kaggle files

```bash
python -m pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('brains14482/nba-playbyplay-and-shotdetails-data-19962021'))"
```

Place or point to `nbastats_2015.csv` through `nbastats_2024.csv`, verify them
against `manifests/NBA_RAW_DATA_SHA256.csv`, then run:

```bash
python experiments/nba/scripts/build_nba_five_var_extension.py \
  --dataset-dir PATH_TO_VERSION_8 \
  --processed-dir scratch/nba/processed \
  --output-dir scratch/nba/output \
  --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25
```

Compare the rebuilt `expanded_team_quarter_{season}.csv` files against
`data/nba/processed_team_quarter/`, which is what `scripts/reproduce_nba.py`
consumes.

## Direct use of the historical scripts

`scripts/reproduce_nba.py` calls the historical scripts unmodified; it only
supplies the directory layout they expect. To drive them yourself, note that
the estimator reads its input from
`<workspace>/outputs/expanded_candidate_study/data_team_loose/expanded_team_quarter_{season}.csv`,
which is the original working layout rather than the repository layout. Stage
the committed CSVs there first, then:

```bash
.venv-nba/Scripts/python experiments/nba/scripts/run_expanded_candidate_optimization.py foul_leaves_5_team \
  --workspace scratch/nba \
  --core experiments/nba/core/d.py \
  --outputs-dir scratch/nba/fits \
  --starts 2 --maxiter 250 --workers 4 \
  --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25
```

`foul_leaves_5_team` is the manuscript's hypothesis: variables `FOUL`, `FTA`,
`FTM`, `PERS_FOUL_DRAWN`, `LOOSE_BALL_FOUL_DRAWN` against the four reference
edges of Figure 5. The other hypotheses in
`experiments/nba/scripts/scan_expanded_candidates_moment.py` are exploratory and
are not reported in the paper.

Two path caveats apply when calling the historical scripts directly:

* `run_expanded_candidate_optimization.py` is the one script whose default was
  path-normalized for this repository. Its `--core` default was changed from an
  absolute path on the original machine to
  `experiments/nba/core/d.py`. The file is byte-identical to the core
  recorded in every committed `data/nba/graph_json/*.json`. No other historical
  script was edited.
* `summarize_nba_extension.py` reads the raw Kaggle files and re-hashes the
  absolute input path recorded inside each frozen `graph_result.json`. Both
  point at the original machine, so the script only runs where that layout
  exists. `scripts/reproduce_nba.py` therefore derives the graph-recovery
  table and the figure inputs by importing that script's metric functions
  rather than by invoking it.
