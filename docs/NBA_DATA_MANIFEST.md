# NBA data manifest

## Committed processed data

| Season | Rows |
|---|---:|
| 2015-16 | 9,840 |
| 2016-17 | 9,840 |
| 2017-18 | 9,840 |
| 2018-19 | 9,840 |
| 2019-20 | 8,448 |
| 2020-21 | 8,640 |
| 2021-22 | 9,840 |
| 2022-23 | 9,840 |
| 2023-24 | 9,840 |
| 2024-25 | 9,840 |
| **Total** | **95,808** |

Processed filenames, byte sizes, and SHA-256 hashes are in `manifests/NBA_DATA_SHA256.csv` and the global `manifests/SHA256SUMS.csv`.

## Raw files not committed

The raw version-8 files total approximately 984 MiB. Their sizes and SHA-256 values are in `manifests/NBA_RAW_DATA_SHA256.csv`. They are omitted to avoid duplicating the upstream dataset; users should obtain them from Kaggle under the dataset's Apache-2.0 terms.

The aggregation audit checks season labels, period range 1–4, nonnegative counts, and `FTM <= FTA` for every row.
