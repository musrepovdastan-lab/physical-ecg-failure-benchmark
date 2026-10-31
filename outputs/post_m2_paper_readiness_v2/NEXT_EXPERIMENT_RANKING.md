# Next-experiment ranking

| Rank | Addition | Scientific value | Compute cost | Publication impact | Recommendation |
|---:|---|---|---|---|---|
| 1 | Replicate with a substantially different ECG architecture | High | Moderate | High | **Workshop next experiment** |
| 2 | Real recorded electrode-reversal/misplacement data | Very high | High data/coordination cost | Very high | **Stronger main-conference option** |
| 3 | RA–LA and LA–LL frozen-model evaluation | Moderate | Low | Moderate | Diagnostic secondary analysis |
| 4 | Calibration and sensitivity/specificity analysis | Moderate | Low | Moderate | Report alongside primary work |
| 5 | Ningbo random-masking and combined augmentation arms | Moderate | Moderate | Low–moderate | Secondary; M2 deliberately froze only the primary arm |

The single recommended next workshop experiment is a three-seed replication with one substantially different ECG architecture under the unchanged physics-aware protocol and the same clean/RA–LL/mask evaluation. The stronger main-conference option is validation on real recorded reversal or misplacement data.

Indicative cost: the architecture replication is approximately 6–12 RTX-5070 GPU-hours plus 1–2 wall-clock days including the fixed-condition evaluation/bootstrap; real-data validation is likely 2–4 GPU-hours once curated, but several weeks of data access, adjudication and governance work.
