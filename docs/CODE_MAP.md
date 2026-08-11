# Code map

What each program reads, writes, and how they chain. The numerical cores and
the NBA scripts are byte-identical to the code that produced the committed
results.

## Entry points

| Program | Environment | Reads | Writes |
|---|---|---|---|
| `scripts/build_paper_objects.py` | none | `manifests/PAPER_OBJECTS.csv`, produced figures | `outputs/` |
| `scripts/verify_frozen_results.py` | none | `manifests/SHA256SUMS.csv`, committed results | stdout |
| `scripts/check_environment.py` | none | `requirements-*.txt` | stdout |
| `scripts/build_manifests.py` | none | source, data and results | `manifests/SHA256SUMS.csv` |
| `scripts/run_all_poisson.py` | simulation | `config/all_poisson_final.json`, `src/` | `results/all_poisson/` |
| `scripts/summarize_all_poisson.py` | simulation | `results/all_poisson/formal_cells/` | `results/all_poisson/{raw,summaries,metadata}` |
| `scripts/plot_figure4.py` | simulation | `results/all_poisson/summaries/` | `final_figures/` |
| `scripts/run_mixed_family.py` | simulation | `config/mixed_family_final.json`, `src/` | `results/mixed_family/` |
| `scripts/summarize_mixed_family.py` | simulation | `results/mixed_family/formal_cells/` | `results/mixed_family/{raw,summaries}` |
| `scripts/plot_mixed_family.py` | simulation | `results/mixed_family/summaries/` | `final_figures/`, `results/mixed_family/plotdata/` |
| `scripts/reproduce_nba.py` | nba | `data/nba/processed_team_quarter/` | work dir, then `outputs/` via `build_paper_objects.py` |

`final_figures/` and work directories are ignored by Git. `outputs/` is
derived: `manifests/PAPER_OBJECTS.csv` records its digests, and
`manifests/SHA256SUMS.csv` covers source, data and results.

## Chains

**Figure 1 (all-Poisson).** `run_all_poisson.py` generates every dataset from
master seed `20260622` and fits the seven methods, writing one directory per
physical cell. `summarize_all_poisson.py` consolidates those into the
per-replication table, the 240-cell summary and a validation report; it refuses
to emit a summary if any cell is incomplete. `plot_figure4.py` draws from the
summary and refuses to draw unless the validation report is `passed`.

**Figures 2–4 (mixed-family setting).** `run_mixed_family.py` →
`summarize_mixed_family.py` → `plot_mixed_family.py`. The plotting step also
writes `plotdata/`, the values behind each curve.

**Figures 5–6 and Table 3 (NBA).** `reproduce_nba.py` stages the committed
processed CSVs into the directory layout the historical scripts expect, calls
`run_expanded_candidate_optimization.py` for the `foul_leaves_5_team`
hypothesis, then derives the recovery table and figure inputs by importing the
metric functions of `summarize_nba_extension.py` rather than invoking it. See
`docs/NBA_REPRODUCTION.md` for why that indirection exists.

## Numerical cores

`src/d.py` and `experiments/nba/core/d.py` are two builds of the same program.
Both implement the PT-SEM likelihood, the plug-in BIC, exact subset dynamic
programming, and adapters for the external baselines. They differ in the
negative-binomial branch of `convolution_loglik`:

- `src/d.py` (`70da3d96…`, `ptsem_final_nb_exact_v2`) accumulates the finite
  Delaporte convolution in log space and raises on non-finite values.
- `experiments/nba/core/d.py` (`d14c13f5…`) uses the earlier `hyp1f1`
  expression.

Each is used by the line whose committed results record it;
`verify_frozen_results.py --section cores` enforces this.

## Names that differ from the paper

Program output names differ from the paper's numbering.
`manifests/PAPER_OBJECTS.csv` holds the mapping and `outputs/` applies it;
`docs/PAPER_RESULT_MANIFEST.md` is the readable form.

`results/nba/tables/table2_nba_graph_recovery.csv` is the paper's Table 3. The
`foul_leaves_5_team` hypothesis denotes the five variables and four reference
edges of Figure 5.

## Conventions

Entry points are argparse programs. Programs that consume a validated artifact
check its report first and exit non-zero if it is not `passed`. Refit entry
points check the interpreter against their pinned profile before running.
