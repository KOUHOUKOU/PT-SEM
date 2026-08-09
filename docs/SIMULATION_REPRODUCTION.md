# Simulation reproduction (Figures 1–4)

All simulated data is generated from a locked master seed, so nothing needs to
be downloaded. Use the simulation environment (`docs/ENVIRONMENT.md`).

## Redrawing the figures from committed results

Seconds, no refitting:

```bash
.venv-simulation/Scripts/python scripts/plot_figure4.py
.venv-simulation/Scripts/python scripts/plot_mixed_family.py
python scripts/build_paper_objects.py
```

`plot_figure4.py` draws manuscript Figure 1; `plot_mixed_family.py` draws
Figures 2–4. Both write into `final_figures/`, and `build_paper_objects.py`
promotes them into `outputs/` under their manuscript names and verifies each
one against the manuscript. Both plotting programs refuse to draw unless the
validation report for their inputs says `passed`.

## Refitting from zero

Both suites are `R = 100` at master seed `20260622` over 36 physical cells.

```bash
.venv-simulation/Scripts/python scripts/run_all_poisson.py --workers 12
.venv-simulation/Scripts/python scripts/summarize_all_poisson.py

.venv-simulation/Scripts/python scripts/run_mixed_family.py --workers 12
.venv-simulation/Scripts/python scripts/summarize_mixed_family.py
```

Recorded runtimes on 24 logical cores with 12 workers: the all-Poisson suite
took 3 h 13 min, the mixed-family suite 3 h 38 min. Both support checkpoint and
resume — rerunning the same command after an interruption continues rather than
restarting, and the package records identical non-runtime output for 1, 4, 8 and
12 workers.

`run_all_poisson.py --preflight-only` exercises the whole path at trivial scale
and is worth running first; it finishes in under a minute.

## External baselines

PB-SCM and PB-SCM-PGF have no upstream license and are not redistributed here.
The core reports the exact clone command when they are missing. See
`docs/THIRD_PARTY.md`. Without them, the two PB-SCM curves cannot be recomputed;
everything else can.

## Design

`config/mixed_family_final.json` and `config/all_poisson_final.json` are the
locked designs. Their sweep grids are what manuscript Table 2 states:

- dimension `d ∈ {4,…,10}` at `N = 3200`, average indegree `1.5`
- sample size `N ∈ {100, 200, 400, 800, 1600, 3200, 6400, 10000}` at `d = 8`
- average indegree `∈ {1.0, 1.5, 2.0, 2.5, 3.0}` at `d = 8`, `N = 3200`

restricted uses `α ~ Uniform(0.15, 0.85)`, extended uses `α ~ Uniform(0.2, 2.0)`.
`scripts/verify_frozen_results.py` checks the executed grid against this.

## Seeds

`results/*/metadata/seeds.csv` records the data seed and method seed of every
one of the 24,000 rows in each suite. Seeds derive deterministically from the
master seed and the configuration and are independent of platform, so they can
be regenerated and compared before spending any compute.
