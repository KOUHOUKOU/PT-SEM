# Known limitations and unresolved provenance

## Historical Git identity

The recovered BEST_PTSEM and Basketball source trees were not Git repositories. There is no earlier commit SHA or branch to report. This is a provenance fact, not an omitted field.

## Sample-size discrepancy

The actual mixed-family run and Figure 1–3 plot data include `N=2400`. The current manuscript's Table 2 lists `100, 200, 400, 800, 1600, 3200, 6400, 10000` and omits 2400. The repository preserves the executed grid and does not delete the point to force agreement. This discrepancy is classified `UNRESOLVED` pending an author/editor decision about correcting the manuscript table.

## Negative-binomial thinning convolution

After the paper artifacts were produced, a numerical audit established that the historical `hyp1f1` expression in the NB thinning convolution was invalid on part of the parameter domain. This can affect the Proposed and Greedy BIC scores whenever NB is a candidate family, including all-Poisson truth because the fitted library still considers NB.

The exact historical code and paper results are retained to make the manuscript reproducible. Corrected calculations are deliberately kept in `audit/corrected_nb_convolution/` as audit evidence and are not presented as replacements for Figures 1–4. Any future scientific revision should be versioned as a distinct result release.

## Third-party licenses

PB-SCM and PB-SCM-PGF had no upstream license at freeze time. Their source is therefore not redistributed. Reproduction depends on the upstream repositories remaining available or on users obtaining lawful copies.
