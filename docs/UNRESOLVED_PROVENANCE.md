# Known limitations and provenance

## Negative-binomial thinning convolution

An earlier numerical core evaluated the negative-binomial branch of the
Poisson-thinning convolution through a `hyp1f1` expression. The identity is
correct, but SciPy does not evaluate it reliably when its second parameter
`1 − r − y` is negative: at `r = 1.48698, p = 0.22583, μ = 65.54` the returned
value is negative at `y = 100`, which is impossible for an all-positive series,
and positive but wrong at `y = 140`. The old code fell back only on non-finite
results, so finite wrong values passed; summed over `y = 0…400` the implied
probability mass was `1.66e26` rather than 1.

Figures 1–4 use the corrected core `src/d.py` (`70da3d96…`,
`ptsem_final_nb_exact_v2`), which accumulates the finite convolution in log
space, raises on non-finite values, and normalises to `1.000000000000`. The
audit in `audit/corrected_nb_convolution/` covers 7,540 NB cases with zero
failures.

Figures 5–6 and Table 3 use `experiments/nba/core/d.py` (`d14c13f5…`), the core
recorded in every `data/nba/graph_json/*.json`. Recomputing all 4,800 NBA local
scores under both cores gives a maximum absolute BIC difference of `8.9e-05`,
against a margin of `9.19` between the best and second-best DAG in every season;
the selected graph is unaffected.

## Third-party baselines

PB-SCM and PB-SCM-PGF carry no upstream license, so their source is not
redistributed. Reproducing their two curves in Figure 1 requires obtaining them
separately; see `docs/THIRD_PARTY.md`. Every other curve is reproducible from
this repository alone.

## Upstream NBA data

The raw Kaggle play-by-play files are not redistributed. The processed
team-quarter derivatives consumed by the analysis are committed and
hash-frozen; rebuilding them from raw requires the upstream dataset. See
`provenance/DATASET_LOCATION.md`.

## Absolute paths inside frozen run records

`data/nba/graph_json/*.json`, `results/*/metadata/formal_run_plan.json` and the
convolution audit report record the absolute paths of the machine that produced
them. These files are shipped as the runs wrote them; rewriting a field would
change their digests and break the correspondence with
`manifests/SHA256SUMS.csv`. The identity used for verification is the recorded
SHA-256 of the numerical core, which
`scripts/verify_frozen_results.py --section cores` checks.

## Historical version control

The source trees this work descends from were not Git repositories, so no
earlier commit identifier can be reported.
