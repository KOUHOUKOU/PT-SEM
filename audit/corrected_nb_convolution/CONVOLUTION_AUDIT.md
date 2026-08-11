# PT-SEM convolution-path audit

Status: **PASSED on the locked `src/d.py` exact-v2 core.**

The current audited core SHA256 is
`70da3d96d4013270de44e0dce1653cc27e6e2ab6582fa4b124ac962e1ebdb020`.

The audit is read-only with respect to the project and frozen snapshot.  It
enumerated all convolution implementations found by a project-wide search:

1. `src/d.py::convolution_loglik`, whose NB branch is the corrected exact
   finite log-space recurrence;
2. the NBA real-study `exact_convolution_loglik` wrapper, which implements a
   grouped Geometric convolution and delegates all other families to the core.

## Results

| Family | Formula/stress cases | Failures | Maximum absolute log-PMF difference |
|---|---:|---:|---:|
| Poisson | 1,270 | 0 | 1.82e-12 |
| NB, corrected v2 | 7,540 | 0 | 1.19e-7* |
| ZIP | 3,800 | 0 | 1.82e-12 |
| Geometric | 1,160 | 0 | 3.64e-12 |
| Binomial | 2,810 | 0 | 3.64e-12 |
| Bernoulli | 1,270 | 0 | 3.64e-12 |

`*` The maximum is one floating-point ULP for a deliberately extreme log-PMF
near -7.6e8.  At parameter combinations arising in saved local-score tables,
the corrected NB maximum was 9.35e-13.  High-precision Decimal checks at the
moment-inversion Poisson boundary r=1e8 showed v2 errors of approximately
1--2e-9, versus approximately 3.2e-7 for v1.

Every probability-mass normalization check passed.  The largest mass error
was 2.06e-10 for a heavy-tailed Geometric case with p=0.005.  Vector likelihood
versus summed scalar likelihood checks passed for all six families.

The NBA wrapper passed 4,200 cases across all six families, with maximum
absolute vector log-likelihood error 3.27e-10.  Its NB path is safe only when
the delegated core is `nb_exact_convolution_v2`; the frozen core remains
invalid.

## Known failed control

The frozen NB path remains a confirmed failure.  At r=1.48698, p=0.22583:

- thinning mean 65.54: frozen PMF mass 1.66e26, corrected mass approximately 1;
- thinning mean 76.83: frozen PMF mass 1.07e32, corrected mass approximately 1.

This failed control is retained intentionally so a future audit cannot pass by
accidentally testing only benign inputs.

## Scope limitation

Passing this audit establishes the convolution layer, family parameterization,
and NBA convolution wrapper after the NB v2 substitution.  It does not by
itself prove the absence of defects in DAG generation, moment estimation,
graph search, external baselines, metrics, aggregation, or plotting; those
remain separate provenance and validation layers.

Primary report: `audit/corrected_nb_convolution/convolution_audit_report.json`.
Reproduction command:

```powershell
python scripts/verify_frozen_results.py --section cores
```
