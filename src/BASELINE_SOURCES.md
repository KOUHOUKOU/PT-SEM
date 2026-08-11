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
- The upstream repository currently declares no software license. The pinned
  files required for execution are copied unchanged into this isolated package
  under `src/external/PBSCM`; their inclusion here is for scientific
  reproducibility and does not assert a license grant.

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
  included in the restricted regime. The pinned files required for execution
  are copied unchanged under `src/external/PBSCM_PGF`; the public repository
  currently has no declared software license.

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
