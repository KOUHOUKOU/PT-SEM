# Third-party software

The formal comparison uses external implementations loaded from `src/external/`:

| Method | Repository | Frozen commit | Redistribution |
|---|---|---|---|
| PB-SCM | upstream PBSCM repository documented in `src/BASELINE_SOURCES.md` | `08ba9ba806eacd8162f5f907c941d946a0ce69f0` | excluded; no upstream license found |
| PB-SCM-PGF | upstream PBSCM-PGF repository documented in `src/BASELINE_SOURCES.md` | `5fdea2b7e1b796fd0b48c6e0b3be51959b4fa2c1` | excluded; no upstream license found |

`src/d.py` loads them from an ignored `src/external/` directory and prints the exact clone command when they are absent. They are not covered by this repository's MIT license. `causal-learn`, SciPy, NumPy, pandas, statsmodels, and Matplotlib are installed from their normal package distributions under their respective licenses.
