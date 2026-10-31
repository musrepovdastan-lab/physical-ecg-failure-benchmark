# Mechanism audit

## What the cached predictions directly support

Across both datasets, the matched contrast is positive for every endpoint and every seed. The contrast is large for TWA and STach, while SB is small in Chapman and modest in Ningbo. The table below is the synchronized bootstrap aggregate, not an average of prediction vectors.

| Dataset | Endpoint | Contrast | 95% CI |
|---|---:|---:|---:|
| Chapman | TWA | 0.246 | [0.210, 0.280] |
| Chapman | SB | 0.005 | [0.003, 0.007] |
| Chapman | STach | 0.227 | [0.187, 0.273] |
| Ningbo | TWA | 0.331 | [0.311, 0.350] |
| Ningbo | SB | 0.038 | [0.032, 0.045] |
| Ningbo | STach | 0.271 | [0.250, 0.294] |

The cached prediction analysis (`MECHANISM_PREDICTION_SUMMARY.csv`) separates true class and clean-model correctness, reports clean-to-fault probability shifts, and counts TP→FN, TN→FP, FN→TP and FP→TN transitions at the validation-frozen threshold. The mean-mask probability in that file is the descriptive mean of the 792 cached mask prediction vectors; it is not used for the prespecified AUPRC contrast or any new inference.

## Interpretation labels

- **Directly supported:** RA–LL reversal produces a substantially larger AUPRC loss than the mean five-displayed-lead masking consequence in all TWA seed/dataset combinations, and also in all STach combinations. SB is affected less, especially in Chapman.
- **Directly supported:** the STach effect is not reconciled by the fact that the physiological cardiac rate should be unchanged under an ideal terminal permutation; cached STach probability shifts and error transitions demonstrate model-output sensitivity to the transformed waveform.
- **Consistent with evidence:** the model uses morphology and/or lead configuration features correlated with STach beyond a rate estimate, because a terminal permutation changes its outputs despite preserving the underlying beat timing. The cached predictions cannot identify which features.
- **Speculative:** reliance on a particular limb-lead morphology, learned lead-order dependence, or label/recording confounding could contribute. These are hypotheses only and require prespecified targeted tests.

Ningbo age/sex fields, where present in the frozen canonical record table, were summarized only as descriptive strata in `MECHANISM_STRATA_SUMMARY.csv`; they do not establish effect modification or causation.
