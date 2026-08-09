# Simulation reproduction

## Fast paper-figure path

```bash
python scripts/verify_frozen_results.py
python scripts/generate_figures.py --only simulation
```

This reads only committed result and plot-data files and normally finishes within a minute.

## Formal mixed-family rerun

The historical implementation is intentionally unmodified in `experiments/simulation/legacy/`.

```bash
python experiments/simulation/legacy/setup_external_baselines.py
python experiments/simulation/legacy/ptsem_six_family_final_suite.py \
  --R 100 --workers 12 --seed 20260622 \
  --regime both --sweeps d N kbar \
  --d-values 4 5 6 7 8 9 10 \
  --N-values 100 200 400 800 1600 2400 3200 6400 10000 \
  --kbar-values 1 1.5 2 2.5 3 \
  --results-root scratch/mixed_family_rerun
```

Use `--smoke` before a formal run. The original run used 12 workers and took about five hours on its recorded Windows workstation; runtime will vary substantially. Third-party baseline checkout requirements are in `THIRD_PARTY.md`.

## All-Poisson rerun

The original all-Poisson programs use snapshot-relative paths. The portable entry point prepares that layout in `scratch/` without starting an expensive computation unless `--execute` is supplied:

```bash
python scripts/run_all_poisson.py --setup-external --workspace scratch/all_poisson_rerun
python scripts/run_all_poisson.py --execute --workspace scratch/all_poisson_rerun -- --workers 12 --R 100
```

For result-only regeneration:

```bash
python scripts/generate_figures.py --only all-poisson
```

The committed formal output contains 24,000 rows across 4,000 datasets and 240 method-setting cells. A full rerun is expensive; `data/all_poisson/metadata/validation_report.json` records the completed validation.
