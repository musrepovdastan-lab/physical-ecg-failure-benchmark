# Revised contribution and positioning

## Proposed working title

**When limb-cable reversals disrupt ECG classifiers: a matched stress-test and comparison of protection strategies**

This title states the tested phenomenon and study design without implying priority, superiority, or clinical validation. An even narrower version, if the final paper remains focused on TWA, is: **A matched evaluation of simulated limb-cable reversal and missing-lead stress tests for ECG classifiers**.

## Contribution statement

We report a bounded evaluation of diagnostic classifiers under algebraically specified, simulated limb-terminal permutations and compare those inputs with defined missing-lead conditions, reversal-aware training, chest-only inputs, and a published detector/correction pipeline. Across the evaluated ResNet settings, physics-aware training substantially improved TWA ranking under simulated RA–LL relative to ordinary clean training, while corrected mask comparisons and chest-only controls show that conclusions depend on the fault/control definition and do not establish a consistent advantage over simpler alternatives. A patient-disjoint HEEDB subset supplies an additional transport setting using physician-code proxy outcomes; it does not represent an independent clinical center or verified cable failures. The work is therefore an evaluation and comparative evidence package for specified simulations, endpoints, datasets and models, rather than a new general reversal-robust architecture or proof of clinical benefit.

## Recommended article framing

The closest prior article, Cao et al. (ICASSP 2020), already proposes a clean-trained multi-scale attention architecture and evaluates it under simulated RA–LL and other limb reversals. Lee et al. (2025) further report diagnostic-model robustness to 12 simulated reversal configurations for LVSD. The present paper must not claim that diagnostic ECG robustness to lead permutations, reversal simulation, or model training for robustness is itself new.

What remains defensible is a narrower comparison across the project's **specific** experimental choices: corrected exact-mask and mask-panel controls, a predeclared physical-swap family, reversal-aware training, chest-only input, selected architectures, and the published Huang detector/correction chain. The distinct contribution is the empirical comparison and the boundaries exposed by it, not an unqualified new mitigation algorithm. The most important result is that augmentation improves the tested clean-trained 12-lead model under RA–LL, while its advantage over a chest-only classifier is not established by the paired AUPRC intervals and the Huang chain remains a strong exploratory comparator.

## Statements to remove or narrow

- Remove any “first,” “novel reversal-robust method,” “state of the art,” or universal robustness claim.
- Replace “missing-lead tests miss physical failures” with the exact reported comparisons and describe missingness as a different corruption mechanism, not a matched physical effect.
- Replace “augmentation outperforms alternatives” with endpoint- and dataset-specific point estimates and intervals; do not claim equivalence when an interval crosses zero.
- Keep Transformer outcomes architecture-specific and preserve the M3 failure.
- Call HEEDB outcomes proxy labels and the Huang comparison post-hoc/exploratory. Do not convert score recovery to patient detection or utility.
- State clearly that the project faults are simulated, not adjudicated real acquisition failures.
