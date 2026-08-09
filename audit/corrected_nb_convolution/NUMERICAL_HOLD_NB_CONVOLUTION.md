# Numerical hold: NB--Poisson convolution

Status: **DO NOT PROMOTE PROPOSED DP-BIC OR GREEDY-BIC RESULTS TO THE PAPER**
until the exact-NB rerun and comparison are complete.

The frozen source remains unchanged.  Its optimized NB--Poisson likelihood
uses `scipy.special.hyp1f1`; in the extended/high-count diagnostic this call
can return finite but grossly incorrect values, so the existing non-finite
fallback is not activated.  The affected likelihood can create spuriously low
NB BIC scores.

Scope:

- Potentially affected: Proposed DP-BIC and Greedy-BIC whenever NB is among
  the candidate residual families.
- Not affected by this defect: Poisson-only DP, which never evaluates the NB
  likelihood.
- The snapshot candidate
  `final_figures/fig4_all_poisson_library_cost_final.pdf` has not overwritten
  the paper PDF and must not be copied there while this hold is active.
- The failed R=10 pilot remains quarantined under
  `new_results/all_poisson/diagnostics/extended_convergence_pilot_R10/`.

Correction candidate (v1 superseded by v2):

- `scripts/nb_exact_convolution_v2.py` forces every NB--Poisson likelihood to
  use its complete finite convolution accumulated in log space and avoids the
  large-r cancellation found during the full convolution audit.
- `scripts/nb_exact_convolution_v1.py` is retained for provenance only and
  must not be used for final reruns.
- `scripts/validate_nb_exact_convolution_v1.py` validates the correction
  against the frozen per-observation exact reference and probability-mass
  normalization.
- Validation output:
  `new_results/all_poisson/diagnostics/nb_exact_convolution_v1_validation.json`.

This hold may be removed only after the same isolated R=10 seeds are rerun
with the versioned exact implementation and their local scores are audited.
