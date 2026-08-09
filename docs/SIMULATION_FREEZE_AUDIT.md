# Simulation freeze audit

## Source recovery

The authoritative historical simulation snapshot was recovered from `BEST_PTSEM/reproducibility/jmlr_final_snapshot_20260721`. The active result directories were traced back to `BEST_PTSEM/ptsem_six_family_final_results`. The snapshot is dated 2026-07-21 and records a completed formal run dated 2026-06-24.

Neither the snapshot nor either parent project contained Git metadata. `SOURCE_STATE.txt` explicitly records this fact. Consequently, no historical commit SHA, tag, or branch exists to recover; inventing one would weaken provenance. This public repository's first commit is therefore a packaging commit, while file hashes preserve the earlier state.

The snapshot README claimed that a root manifest and SHA256 file existed, but neither was present at the recovered root. The present repository repairs that packaging omission by generating `manifests/SHA256SUMS.csv`; it does not rewrite the historical files.

## Frozen formal run

- suite version: `six_family_final_v1`
- status: complete
- replications: 100 per cell
- master seed: 20260622
- workers used: 12
- candidate families: Poisson, NB, ZIP, Geometric, Binomial, Bernoulli
- DAG construction: exact edge count, maximum five parents
- dimension sweep: 4, 5, 6, 7, 8, 9, 10
- sample-size sweep actually run: 100, 200, 400, 800, 1600, 2400, 3200, 6400, 10000
- average-indegree sweep: 1.0, 1.5, 2.0, 2.5, 3.0
- anchor: d=8, N=3200, average indegree=1.5
- restricted coefficients: Uniform(0.15, 0.85)
- extended coefficients: Uniform(0.2, 2.0)
- true family assignment: IID uniform over the six families

The manifest reports 38 completed physical cells and 2,900 pairing checks with zero failures. The committed consolidated CSVs contain the raw method-level rows used for Figures 1–3.

## Classification

The code, consolidated results, plot data, and manuscript PDFs are `VERIFIED EXACT` by hash and lineage. A full expensive rerun was not performed during repository packaging. The sample-size-list discrepancy and later NB-convolution defect are disclosed in `UNRESOLVED_PROVENANCE.md`.
