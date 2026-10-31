# Amendment 010 — Huang clean records also pass through predicted correction

Before final metric tables were produced, input-chain review found that the first full-cohort Huang clean predictions skipped the detector's predicted inverse. That excluded detector false corrections on clean ECGs and did not match the fixed pipeline used for clean A/B calibration. The metric aggregation was interrupted before any final tables or intervals were written.

The six provisional Huang clean prediction files and the two completion manifests are archived under `failed_attempts/huang_clean_without_predicted_inverse/` and bound in `FAILED_ATTEMPT_009.json`. The evaluation now applies predicted-class correction to the same incoming clean raw ECG for all four conditions, including clean. It will recompute only those six affected clean predictions and reuse all other predictions and frozen locks.
