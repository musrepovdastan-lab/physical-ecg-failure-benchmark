# FINAL MULTIFAULT DECISION

Physics-aware training and the published detect-and-correct pipeline are not statistically distinguishable on worst-fault TWA AUPRC; this is not an equivalence claim.
Yes, the inferential answer is consistent across all three faults: every 95% interval includes zero. The point estimates do not all favor the same arm, and interval inclusion is not proof of equivalence.

| TWA AUPRC endpoint | Unprotected clean ResNet | Physics-aware | Huang correct | Physics minus Huang 95% CI |
|---|---:|---:|---:|---:|
| RA-LA | 0.2613 | 0.6694 | 0.6652 | [-0.0113, +0.0204] |
| RA-LL | 0.1822 | 0.6665 | 0.6636 | [-0.0133, +0.0187] |
| LA-LL | 0.6338 | 0.6691 | 0.6735 | [-0.0194, +0.0098] |
| Worst fault | 0.1822 | 0.6665 | 0.6636 | [-0.0129, +0.0169] |

## Physics-aware versus Huang by fault

- RA_LA_reversal: delta +0.0041, 95% CI [-0.0113, +0.0204] (no_statistically_resolved_difference).
- RA_LL_reversal: delta +0.0029, 95% CI [-0.0133, +0.0187] (no_statistically_resolved_difference).
- LA_LL_reversal: delta -0.0044, 95% CI [-0.0194, +0.0098] (no_statistically_resolved_difference).
- Worst-fault endpoint: delta +0.0029, 95% CI [-0.0129, +0.0169].

## Hardest case and clean performance

The lowest protected-method mean AUPRC is TWA under RA_LL_reversal (0.6651).
Worst-fault AUPRC by diagnosis is: physics-aware TWA 0.6665, SB 0.9954, STach 0.9843; Huang TWA 0.6636, SB 0.9960, STach 0.9780.
For TWA clean AUPRC, physics-aware minus clean ResNet is +0.0108; Huang correction minus clean ResNet is -0.0007.
The detector leaves 98.51% of clean ECGs untouched and falsely corrects 1.49%.
No material clean rank-performance damage is visible in the point estimates. Physics-aware slightly improves clean AUPRC for all three diagnoses but lowers fixed-threshold specificity by at most 0.0045; Huang changes clean AUPRC by -0.0007 (TWA), approximately 0 (SB), and -0.0014 (STach). These clean comparisons are descriptive because no paired clean-performance hypothesis tests were prespecified.

## Research direction

This audit alone is not enough to justify developing a novel worst-case/consistency training method. It establishes the multifault operational baseline and identifies residual weaknesses, but a new method still needs a prespecified hypothesis, validation-only development, and an independent held-out evaluation.

## Unsupported claims

This is not a reproduction of Huang et al.'s paper scores, not evidence of SOTA superiority, not a clinical-prevalence study, and not evidence for real electrode errors, artifacts, devices, populations, or fault types beyond the exact simulated terminal swaps tested here. Detector calibration, non-argmax policies, Huang-model adaptation, and prospective workflow utility remain untested.

The oracle arm uses the true fault only as an implementation upper bound and is not deployable.
