# Computational environment

The formal mixed-family manifest records Python 3.11.9 on Windows and the following method-critical versions:

| Package | Version |
|---|---:|
| numpy | 1.26.4 |
| pandas | 3.0.3 |
| scipy | 1.13.0 |
| statsmodels | 0.14.6 |
| causal-learn | 0.1.4.7 |
| matplotlib | 3.10.9 |
| scikit-learn | 1.9.0 |
| KDEpy | 1.1.4 |

`requirements.txt` pins the reproduction-critical environment. `environment/historical_full_environment.txt` is the broader environment captured later with the 2026-07-21 snapshot; it contains unrelated notebook and data-access packages and reports numpy 2.2.6. The conflict is preserved rather than hidden: the formal run manifest is authoritative for the mixed-family estimator, while the later full freeze is useful for figure/audit utilities.

Multiprocessing scripts set numerical-library thread counts to one before importing NumPy/SciPy to avoid worker oversubscription. Results may vary in runtime across BLAS builds; seeded data and graph metrics should remain stable.
