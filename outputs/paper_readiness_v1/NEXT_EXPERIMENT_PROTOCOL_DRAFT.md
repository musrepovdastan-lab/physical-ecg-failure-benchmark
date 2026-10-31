# Draft protocol: physics-aware mitigation experiment (not executed; corrected)

Chapman and Ningbo test outcomes have already been observed. This is a prespecified follow-up mitigation experiment, not a newly sealed confirmation. Freeze all data, model, augmentation, metric and decision choices before new training.

## Ordered methods and equal augmentation exposure

- **Cached standard clean-training baseline:** 100% clean; reused, not retrained.
- **Primary method — physics-aware augmentation:** 50% clean and 50% uniformly sampled exact terminal reversal from RA–LA, RA–LL and LA–LL. The sole primary hypothesis compares this method against the cached clean baseline.
- **Secondary method — random five-lead masking:** 50% clean and 50% uniformly sampled five-displayed-lead mask using the frozen train-mean sentinel.
- **Secondary method — combined augmentation:** 50% clean, 25% five-lead masking and 25% uniformly sampled exact terminal reversal.

Secondary methods cannot satisfy or replace the primary physics-aware hypothesis.

## Staged execution

### Stage M1 — Chapman development

Reuse the three cached clean baselines (104729, 130363, 155921). Train only the three new augmentation arms for the same three seeds. No test-dependent tuning is allowed.

### Stage M2 — Ningbo follow-up

Apply the unchanged primary physics-aware method to Ningbo only if the prespecified Chapman development gate passes.

## Prespecified primary success gate (TWA)

Physics-aware augmentation versus the cached standard clean baseline succeeds only if all conditions hold: (1) RA–LL TWA AUPRC increases by at least 0.15 absolute in each seed; (2) aggregate matched contrast decreases by at least 0.15 and the synchronized 1,000-replicate 95% upper bound is at most 0.15; (3) mean clean TWA AUPRC declines by no more than 0.02, with no seed worse by more than 0.04; (4) mean 792-mask TWA AUPRC declines by no more than 0.02, with no seed worse by more than 0.04; and (5) all seeds pass the pre-test validation gate (TWA AUROC ≥0.75, AUPRC ≥0.45, finite training/predictions and no leakage).

Report SB and STach, validation-frozen sensitivity/specificity, and 15-bin ECE as secondary safeguards.
