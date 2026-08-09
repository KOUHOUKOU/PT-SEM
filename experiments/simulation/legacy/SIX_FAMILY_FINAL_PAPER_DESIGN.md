# Paper-final six-family PT-SEM sensitivity design

## Scientific question

Can node-wise finite-library BIC recover the DAG, thinning coefficients, and
working exogenous family when every node independently draws its exogenous
law from a heterogeneous six-family library?

## True-family assignment and working library

For every node independently,

`F_v ~ Categorical(1/6, ..., 1/6)`

over `Poisson, NB, ZIP, Geom, Binomial, Bernoulli`.  A graph is not forced to
contain every family.  The candidate working library contains all six
families for every node and every candidate parent set.  The program records
the realized family proportions, full confusion matrices, micro family
accuracy, and macro family accuracy.

Parameter draws are independent and uniform on the declared ranges:

- Poisson lambda: `[2,10]`;
- NB real-valued r: `[2,10]`, p: `[0.15,0.85]`;
- ZIP lambda: `[2,10]`, rho: `[0.15,0.85]`;
- Geometric mean: `[2,10]`;
- Binomial integer n: `{2,...,20}`, p: `[0.15,0.85]`;
- Bernoulli p: `[0.15,0.85]`.

## Alpha regimes

1. **Common:** `alpha ~ Uniform(0.15,0.85)`.  Methods are LibraryDP,
   LibraryGreedy, OracleDP, PC-RCIT, PBSCM, PBSCM_PGF, and PoissonDAG-ODS.
2. **Expanded:** `alpha ~ Uniform(0.2,2.0)`.  PBSCM and PBSCM_PGF are excluded
   because their binomial-thinning model requires coefficients in `[0,1]`.
   LibraryDP, LibraryGreedy, OracleDP, PC-RCIT, and PoissonDAG-ODS remain.

The regimes use the same underlying uniform random coefficient draws after
rescaling, the same DAG edges, the same node families, and the same exogenous
parameters for every matched `(d, replicate)`.

## Three sensitivity sweeps

- d-sweep: `d=4,5,6,7,8,9,10`; `N=3200`; exact average indegree `1.5`.
- N-sweep: `N=100,200,400,800,1600,2400,3200,6400,10000`; `d=8`; exact
  average indegree `1.5`.
- density-sweep: exact average indegree `1.0,1.5,2.0,2.5,3.0`; `d=8`;
  `N=3200`.

All experiments use `R=100`, `max_parents=5`, base seed `20260622`, exact-edge
DAG generation, and 12 replicate-level workers by default.  At the common
anchor `(d=8,N=3200,kbar=1.5)`, all three sweeps reuse the same cell rather
than recomputing nominally identical results.

For the N-sweep, all N values share the same truth for a replicate.  For the
density-sweep, low-density graphs are edge subsets of higher-density graphs;
shared edges retain exactly the same alpha and all node exogenous
specifications remain fixed.  Every method within a replicate receives an
identical copy of the same data.

## Outputs and safeguards

Each sweep writes raw results, summary results, family confusion, family
metrics, family-assignment audit, and figures.  The top-level output contains
an all-sweeps summary, a reproducibility audit, and a `PAPER_FIGURES` folder.

The suite validates all six convolution likelihoods, moment inversions, exact
order-graph DP against exhaustive enumeration, IID family assignment, and
expanded-alpha count generation before simulation.  Cell manifests prevent
mixing incompatible configurations.  Results are checkpointed after every
replicate and resume automatically.  The Windows sleep lock prevents system
sleep while allowing the display to turn off.

Run the formal suite by double-clicking:

`run_six_family_final_suite_overnight.bat`
