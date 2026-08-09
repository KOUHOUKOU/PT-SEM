# External baseline provenance

## PoissonDAG-ODS (GLMLasso)

- Paper: Gunwoong Park and Garvesh Raskutti, *Learning Large-scale Poisson
  DAG Models based on Overdispersion Scoring*, NeurIPS 2015.
- Source: https://pages.cs.wisc.edu/~raskutti/publication/PoissonDAG_nips2015.pdf
- Implementation in `d.py`: all three stages of Algorithm 1 are kept
  separate. Step 1 learns the moral-graph neighborhood with node-wise
  L1-penalized Poisson GLMs and OR-rule symmetrization. Step 2 uses the
  Poisson overdispersion score in equation (4), with the paper's simulation
  cutoff `c0 = 0.005`. Step 3 performs a second L1-Poisson parent-selection
  regression among earlier moral neighbors; it does not merely orient an
  externally supplied skeleton.
- GLMLasso configuration: predictors are standardized as in glmnet, the
  intercept is unpenalized, and `lambda = 0.1` is fixed to the NeurIPS-2015
  simulation value. `statsmodels 0.14.6` supplies the deterministic
  coordinate-descent solver.
- This is an external Poisson-DAG model and is not a fixed-family PT-SEM
  ablation.

## PBSCM

- Paper: Jie Qiao, Yu Xiang, Zhengming Chen, Ruichu Cai, and Zhifeng Hao,
  *Causal Discovery from Poisson Branching Structural Causal Model Using
  High-Order Cumulant with Path Analysis*, AAAI 2024, 38(18), 20524--20531.
- Paper: https://arxiv.org/abs/2403.16523
- Author repository: https://github.com/DMIRLAB-Group/PBSCM
- Pinned source commit: `08ba9ba806eacd8162f5f907c941d946a0ce69f0`
- Adapter: calls the author's `PB_SCM.Hill_Climb_search()` followed by
  `learning_causal_direction()` without replacing either stage by a proxy.
- The upstream repository currently declares no software license. Its source
  remains an intact external Git clone under `external/PBSCM`; it is not copied
  into this project.

## PBSCM-PGF

- Paper: Yu Xiang, Jie Qiao, Zefeng Liang, Zihuai Zeng, Ruichu Cai, and
  Zhifeng Hao, *On the Identifiability of Poisson Branching Structural Causal
  Model Using Probability Generating Function*, NeurIPS 2024 (Spotlight),
  37:11664--11699.
- Paper: https://openreview.net/forum?id=TUwWBLjFk9
- Author repository: https://github.com/DMIRLAB-Group/PBSCM-PGF
- Pinned source commit: `5fdea2b7e1b796fd0b48c6e0b3be51959b4fa2c1`
- Adapter: calls the author's `PBSCM_PGF.learn()` end to end, including its PGF
  rank-bootstrap skeleton and triangle/chain orientation stages. The author's
  analytic NumPy PGF derivative is used in place of its equivalent torch
  automatic derivative, and bootstrap jobs are scheduled serially inside each
  replicate to prevent nested process/thread oversubscription; neither change
  alters the statistical algorithm or number of bootstrap samples.
- Like the cumulant PB-SCM, this method assumes binomial thinning and is only
  included in the common-alpha regime. The public repository currently has no
  declared software license, so its intact source remains an external clone.

## PC-RCIT

- Implementation: `causal-learn` PC-Stable with the scalable randomized
  conditional independence test RCIT, treating high-cardinality counts as
  numeric variables. The test uses `alpha = 0.05`, `approx = "lpd4"`,
  `num_f = 100`, `num_f2 = 5`, and `rcit = True`.
- Random Fourier features use the replicate-specific recorded method seed.
- Raw-count G-squared testing is deliberately not used because the unbounded,
  high-cardinality supports create extremely sparse conditional contingency
  tables. Fisher-Z is not used in the corrected main baseline.
- Package: https://github.com/py-why/causal-learn

## CPCM

- Paper: Juraj Bodik and Valerie Chavez-Demoulin, *Identifiability of causal
  graphs under nonadditive conditionally parametric models*, JMLR 2025.
- Author repository: https://github.com/jurobodik/Causal_CPCM
- Pinned source commit: `2ada90e342d75500494d692fdc86be93044ed659`.
- Adapter: invokes the author's R-only `CPCM_graph_estimate()` with
  `family_of_distributions="Sequential choice"`,
  the author-provided scalable option `greedy_method="edge_greedy"`, and
  `lambda=1`, then reads the returned `bnlearn` arcs.
- Status: excluded from the main Experiment 1 because the paper and local
  benchmarks both show prohibitive scaling at `N=10000, d=8`. The verified
  adapter and `--include-cpcm` remain available for optional appendix runs;
  no Python proxy is used.
