# M3 protocol draft — not approved or executed

## Purpose and necessity under review

M3 would test whether the observed mitigation is architecture-specific. It is necessary only if reviewers require evidence beyond ECGResNet-small; it is not authorized by this draft.

## Proposed design

- One substantially different ECG architecture; three seeds 104729, 130363 and 155921.
- Standard clean-trained baseline and the already frozen 50% clean / 50% exact terminal-reversal physics-aware arm.
- Preserve data splits, endpoint definitions, exact transformations, validation gates, masks, record-level paired bootstrap and no-test-before-freeze rule.
- Primary hypothesis: physics-aware training reduces TWA matched contrast versus its clean-trained counterpart by at least 0.15 with 95% upper CI no greater than 0.15, while clean and mean-mask safeguards match M2.

## Statistical role, cost and approval gate

M3 is confirmatory for architecture generality, not a replacement for M2. Pre-register all architecture/hyperparameter choices before training. Expected cost is 6–12 RTX-5070 GPU-hours and 1–2 wall-clock days. It may begin only after review accepts its necessity, compute allocation, thresholds and role; this document does not authorize execution.
