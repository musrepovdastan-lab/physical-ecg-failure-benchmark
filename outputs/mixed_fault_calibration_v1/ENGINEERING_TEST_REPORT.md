# Engineering test report

All synthetic weighted-mixture checks passed before analysis.

- **q_endpoints:** exact raw clean/reversal metric equivalence
- **weighted_AP_AUROC:** matched sklearn sample_weight reference
- **ties_zero_weights:** finite; zero-mass rows ignored
- **paired_bootstrap:** record multiplicity reused for both copies
- **brier_mixture_identity:** PASS
- **raw_calibrated_threshold_equivalence:** PASS
- **monotone_ranking_and_clipping_ties:** PASS
