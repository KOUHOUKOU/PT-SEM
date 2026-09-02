# Data and external sources

## NBA play-by-play data

The NBA study uses version 8 of Vladislav Shufinskiy's Kaggle dataset *NBA WNBA
play-by-play and shots data* (`brains14482`), released June 26, 2025 and
downloaded May 12, 2026:

https://www.kaggle.com/datasets/brains14482/nba-playbyplay-and-shotdetails-data-19962021/versions/8

The upstream dataset is licensed under Apache 2.0. Its raw `nbastats_2015.csv`
through `nbastats_2024.csv` files are not duplicated in this repository.
Expected byte sizes and SHA-256 values are recorded in
`manifests/NBA_RAW_DATA_SHA256.csv`.

The analysis-ready team-quarter files for seasons 2015-16 through 2024-25 are
committed under `data/nba/processed_team_quarter/`. Their SHA-256 values are in
`manifests/NBA_DATA_SHA256.csv`; the same hashes are bound to the fit records by
`results/nba/input_manifest.json`. The ten files contain 95,808 rows in total.

The aggregation programs are retained as
`experiments/nba/scripts/build_quarter_counts.py` and
`experiments/nba/scripts/build_expanded_team_quarter_fouls.py`.

Suggested citation:

> Shufinskiy, V. (2025). *NBA WNBA play-by-play and shots data* (Version 8)
> [Data set]. Kaggle. Downloaded May 12, 2026.

## External baseline implementations

The baseline fetch specification is `config/external_baselines.json`.

- PB-SCM: https://github.com/DMIRLAB-Group/PBSCM at commit
  `08ba9ba806eacd8162f5f907c941d946a0ce69f0`.
- PB-SCM-PGF: https://github.com/DMIRLAB-Group/PBSCM-PGF at commit
  `5fdea2b7e1b796fd0b48c6e0b3be51959b4fa2c1`.

Neither pinned upstream revision contains a redistribution license. Their
source is therefore fetched locally into the ignored `src/external/` directory
and is not distributed by this repository. `scripts/fetch_external_baselines.py`
checks the repository URL, commit, and required files before use.

The remaining external algorithms use pinned Python packages listed in the two
requirements files. ODS follows Park and Raskutti (NeurIPS 2015); PC-RCIT uses
the `causal-learn` implementation and the settings recorded in the result
metadata.
