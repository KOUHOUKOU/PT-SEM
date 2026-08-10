# Computational environments

The two reproduction lines were computed under different NumPy versions and
require separate environments.

| | `requirements-simulation.txt` | `requirements-nba.txt` |
|---|---|---|
| Covers | Figures 1–4 | Figures 5–6, Table 3 |
| Python | 3.11.9 | 3.11.9 |
| **numpy** | **2.2.6** | **1.26.4** |
| pandas | 3.0.3 | 3.0.3 |
| scipy | 1.13.0 | 1.13.0 |
| matplotlib | 3.10.9 | 3.10.9 |
| scikit-learn, statsmodels, causal-learn, KDEpy, networkx, joblib, threadpoolctl, momentchi2, tqdm | identical | identical |

## Why the split is enforced

Refitting NBA season 2015-16 under the pinned NumPy 1.26.4 reproduced the
committed result with a score difference of 0 and identical BICs at all five
nodes. Under NumPy 2.2.6 the same refit differed by 2.3e-05 and changed one
node's selected exogenous family from Geometric to Poisson.

`scripts/check_environment.py` takes a profile and exits non-zero on a
mismatch; `scripts/reproduce_nba.py` refuses to run in the wrong one:

```bash
python scripts/check_environment.py simulation
python scripts/check_environment.py nba
```

## Creating them

```bash
python -m venv .venv-simulation
.venv-simulation/Scripts/python -m pip install -r requirements-simulation.txt

python -m venv .venv-nba
.venv-nba/Scripts/python -m pip install -r requirements-nba.txt
```

Use `bin/python` instead of `Scripts/python` on Linux and macOS.

## What does not need either environment

`scripts/verify_frozen_results.py`, `scripts/build_paper_objects.py` and the
test suite compare committed bytes and structural invariants.
`requirements-verify.txt` pins the two packages they need; continuous
integration installs only that.

## Threading

The numerical core pins BLAS and OpenMP threads to one per worker through
`threadpoolctl` before fitting. Worker count affects runtime only: the
all-Poisson package records identical non-runtime output for 1, 4, 8 and 12
workers.
