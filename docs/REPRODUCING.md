# Reproducing the paper outputs

## Normal result rebuild

Use Python 3.11 and install the two environments exactly as shown in the root
README. From the repository root, run:

```powershell
.venv-simulation\Scripts\python.exe reproduce.py --nba-python .venv-nba\Scripts\python.exe
```

`reproduce.py` creates a temporary workspace under `work/`, renders Figures
1-4 with the simulation interpreter, renders Figures 5-6 with the NBA
interpreter, and recomputes the NBA structural-recovery table from the
per-season records. It installs only the seven canonical objects under
`outputs/` and removes the temporary workspace.

The manifest uses content-stream SHA-256 for PDFs because PDF creation dates
can change without changing rendered scientific content. The table uses
whole-file SHA-256. A successful run reports:

```text
Figure 1 = PASS
Figure 2 = PASS
Figure 3 = PASS
Figure 4 = PASS
Figure 5 = PASS
Figure 6 = PASS
NBA table = PASS
```

On Linux or macOS, substitute `.venv-simulation/bin/python` and
`.venv-nba/bin/python`.

## Reproduction closure

The simulation closure contains:

- `src/simulation_core.py`, the exact core recorded by the simulation runs;
- `src/all_poisson_framework.py` and `src/mixed_family_framework.py`;
- `config/all_poisson.json` and `config/mixed_family.json`;
- master seed `20260622`, recorded seed tables, raw per-replication results,
  summaries, and validation metadata under `results/`;
- the `run_*`, `summarize_*`, and plotting programs under `scripts/`; and
- `requirements-simulation.txt`.

The NBA closure contains:

- `src/nba_core.py`, the exact core recorded by the NBA fits;
- ten processed season files under `data/nba/processed_team_quarter/`;
- ten optimization records and local-family fit grids under
  `results/nba/fits/`;
- baseline records, seed `2024`, and per-season summaries under `results/nba/`;
- optimization settings `starts=2`, `maxiter=250`, and 12 workers in the fit
  records;
- the analysis and plotting programs under `experiments/nba/scripts/`; and
- `requirements-nba.txt`.

`scripts/check_environment.py simulation` or
`scripts/check_environment.py nba` checks the numerically relevant version
pins for the current interpreter.

## Numerical refits

The scientific run programs and their `--help` interfaces are retained, but a
complete end-to-end refit has not been re-executed as a single public workflow
after packaging. Accordingly, this release does not claim or advertise a
verified `--refit` route. Numerical refits must use the corresponding pinned
environment, the committed configurations and inputs, and the recorded seeds
and settings.

PB-SCM and PB-SCM-PGF source is required only when rerunning those external
baselines. Fetch and verify the pinned commits with:

```powershell
.venv-nba\Scripts\python.exe scripts\fetch_external_baselines.py --fetch --report work\external_baselines.json
```
