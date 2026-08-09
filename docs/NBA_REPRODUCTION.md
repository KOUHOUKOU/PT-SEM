# NBA reproduction

## Result-only verification and figures

```bash
python scripts/verify_frozen_results.py --section nba
python scripts/generate_figures.py --only nba
```

This uses committed processed team-quarter data and fitted result tables.

## From raw Kaggle files

```bash
python -m pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('brains14482/nba-playbyplay-and-shotdetails-data-19962021'))"
```

Place or point to `nbastats_2015.csv` through `nbastats_2024.csv`, verify them against `manifests/NBA_RAW_DATA_SHA256.csv`, then run:

```bash
python experiments/nba/scripts/build_nba_five_var_extension.py \
  --dataset-dir PATH_TO_VERSION_8 \
  --processed-dir scratch/nba/processed \
  --output-dir scratch/nba/output \
  --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25
```

For each season, run the proposed fit with the frozen numerical core:

```bash
python experiments/nba/scripts/run_expanded_candidate_optimization.py hypothesis \
  --workspace scratch/nba \
  --core experiments/simulation/legacy/d.py \
  --outputs-dir scratch/nba/fits \
  --starts 2 --maxiter 250 --workers 4 \
  --seasons 2015-16 2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25
```

Run the fixed five-variable baseline comparison and summarizer using `--help` to supply the same processed and output directories. The original proposed fits took roughly 10–16 minutes per season (about 2.2 hours total sequential runtime recorded in the output tables); baseline runtime is additional. Exact runtime depends on CPU, optimizer, and BLAS.

The full scripts are retained without path-normalizing edits. If a historical script's default assumes the original `Basketball/outputs` layout, pass its path arguments or reproduce that layout under `scratch/`.
