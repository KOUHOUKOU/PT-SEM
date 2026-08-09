# Known limitations and unresolved provenance

## Negative-binomial thinning convolution

An earlier numerical core evaluated the NB branch of the Poisson-thinning
convolution through a `hyp1f1` expression. The identity itself is correct — a
Pochhammer reduction gives exactly the terminating hypergeometric series — but
SciPy cannot evaluate it reliably when its second parameter `1 − r − y` is
negative. At `r = 1.48698, p = 0.22583, μ = 65.54` the returned value becomes
negative at `y = 100`, which is impossible for an all-positive series, and
positive but wrong at `y = 140`. Because the old code only fell back on
non-finite results, the "finite but wrong" mode passed silently: summed over
`y = 0…400` the implied probability mass was `1.66e26` instead of 1.

**Figures 1–4 use the corrected core** (`src/d.py`, `70da3d96…`,
`ptsem_final_nb_exact_v2`). It replaces the hypergeometric call with a finite
log-space recurrence, raises on non-finite values instead of falling back
silently, and normalises to `1.000000000000`. The audit in
`audit/corrected_nb_convolution/` covers 7,540 NB cases with zero failures.

**Figures 5–6 and Table 3 use the historical core** (`experiments/nba/core/d.py`,
`d14c13f5…`), because that is the core recorded inside every committed
`data/nba/graph_json/*.json` and therefore the one behind the published NBA
numbers. The defect was measured on this data rather than assumed away:
recomputing all 4,800 NBA local scores under both cores gives a maximum absolute
BIC difference of `8.9e-05`, against a margin of `9.19` between the best and
second-best DAG in every season. The selected graph, and hence Table 3 and
Figures 5–6, is unaffected at that scale.

## Historical Git identity

The recovered source trees this work descends from were not Git repositories.
There is no earlier commit SHA or branch to report. This is a provenance fact,
not an omitted field.

## Third-party licenses

PB-SCM and PB-SCM-PGF had no upstream license at freeze time, so their source is
not redistributed. Reproducing the two PB-SCM curves in Figure 1 depends on the
upstream repositories remaining available or on obtaining lawful copies; see
`docs/THIRD_PARTY.md`. Every other curve is reproducible from this repository
alone.

## Upstream NBA data

The raw Kaggle play-by-play files are not redistributed. The processed
team-quarter derivatives that the study actually consumes are committed and
hash-frozen, so the NBA results are checkable without them; rebuilding those
derivatives from raw requires the upstream dataset. See
`provenance/DATASET_LOCATION.md`.

## Intermediate all-Poisson results from the superseded server run

Manuscript Figure 1 was, in an earlier manuscript version, produced on a Linux
server under a different pinned environment. That run's intermediate tables were
never downloaded and no longer exist. It is not a gap in this repository: the
figure now shipped was recomputed locally under the environment that produces
Figures 2–4, and both its inputs and its outputs are committed here. The earlier
server figure is superseded and is not part of the manuscript.

## Absolute paths inside frozen run records

`data/nba/graph_json/*.json`, `results/*/metadata/formal_run_plan.json` and the
convolution audit report record the absolute paths of the machine that produced
them. These files are shipped as written by the runs, so the paths were not
rewritten: editing them would change their digests and break the correspondence
with `manifests/SHA256SUMS.csv`. The reproducible identity in those records is
the accompanying SHA-256 of the numerical core, not its path, and
`scripts/verify_frozen_results.py --section cores` checks the digest rather than
the location. Prose documentation carries no absolute paths.
