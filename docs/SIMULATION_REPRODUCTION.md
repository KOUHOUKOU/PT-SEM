# Simulation reproduction (Figures 1–4)

All simulated data is generated from a locked master seed; nothing is
downloaded. Use the simulation environment (`docs/ENVIRONMENT.md`).

## Redrawing the figures from committed results

No refitting is required:

```bash
.venv-simulation/Scripts/python scripts/plot_figure4.py
.venv-simulation/Scripts/python scripts/plot_mixed_family.py
python scripts/build_paper_objects.py
```

`plot_figure4.py` draws Figure 1; `plot_mixed_family.py` draws Figures 2–4.
Both write into `final_figures/`; `build_paper_objects.py` copies them into
`outputs/` under their paper names and verifies each against its expected
digest. Both plotting programs require the validation report for their inputs
to be `passed`.

## Refitting from zero

Both suites are `R = 100` at master seed `20260622` over 36 physical cells.

```bash
.venv-simulation/Scripts/python scripts/run_all_poisson.py --workers 12
.venv-simulation/Scripts/python scripts/summarize_all_poisson.py

.venv-simulation/Scripts/python scripts/run_mixed_family.py --workers 12
.venv-simulation/Scripts/python scripts/summarize_mixed_family.py
```

Runtimes on 24 logical cores with 12 workers: 3 h 13 min (all-Poisson),
3 h 38 min (mixed-family). Both support checkpoint and resume; rerunning the
same command after an interruption continues rather than restarting.
Non-runtime output is identical for 1, 4, 8 and 12 workers.

`run_all_poisson.py --preflight-only` exercises the whole path at trivial
scale in under a minute.

## External baselines

PB-SCM and PB-SCM-PGF have no upstream license and are not redistributed here;
the core reports the clone command when they are missing. See
`docs/THIRD_PARTY.md`. Without them the two PB-SCM curves cannot be recomputed.

## Design

`config/mixed_family_final.json` and `config/all_poisson_final.json` are the
locked designs. Their sweep grids are what manuscript Table 2 states:

- dimension `d ∈ {4,…,10}` at `N = 3200`, average indegree `1.5`
- sample size `N ∈ {100, 200, 400, 800, 1600, 3200, 6400, 10000}` at `d = 8`
- average indegree `∈ {1.0, 1.5, 2.0, 2.5, 3.0}` at `d = 8`, `N = 3200`

The restricted regime uses `α ~ Uniform(0.15, 0.85)`, the extended regime
`α ~ Uniform(0.2, 2.0)`.
`scripts/verify_frozen_results.py` checks the executed grid against this.

## Seeds

`results/*/metadata/seeds.csv` records the data seed and method seed of each of
the 24,000 rows in each suite. Seeds derive deterministically from the master
seed and the configuration and are platform-independent, so they can be
regenerated and compared before running the suites.
