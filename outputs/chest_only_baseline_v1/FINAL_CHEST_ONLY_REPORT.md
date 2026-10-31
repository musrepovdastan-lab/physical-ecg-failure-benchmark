# Chest-only ResNet baseline: Chapman and Ningbo

This report compares six newly trained ECGResNet-small models using only V1-V6 with existing clean-trained and physics-aware 12-lead ResNet comparators. Splits, labels, and training hyperparameters were frozen before fitting. No test-set result selected a checkpoint or parameter.

Frozen protocol SHA-256: `dd5d0a6b3747699c5308b798741c6486eb31dad84422750fd498e5b5efc6bcc2`. The six fits are in `TRAINING_STATUS_BY_DATASET_SEED.csv`; trained weights and resume states are under `models/`.

Chest-only does not receive limb leads, masking, or reversal augmentation. It uses train-derived per-channel V1-V6 normalization. The paired physical reversal conditions leave chest inputs unchanged; exact score invariance was checked in the real inference path.

## TWA primary results

| Dataset | Seed | Model | Condition | AUPRC | AUROC | Brier | ECE-15 | Sensitivity | Specificity | Predicted positive |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| Chapman | 104729 | clean_12lead | clean | 0.6472 | 0.9163 | 0.1185 | 0.1262 | 0.790 | 0.883 | 0.240 |
| Chapman | 104729 | clean_12lead | RA_LL_reversal | 0.2753 | 0.6007 | 0.5839 | 0.6459 | 0.881 | 0.174 | 0.836 |
| Chapman | 104729 | physics_aware_12lead | clean | 0.6740 | 0.9152 | 0.1174 | 0.1269 | 0.764 | 0.880 | 0.238 |
| Chapman | 104729 | physics_aware_12lead | RA_LL_reversal | 0.6384 | 0.8922 | 0.1447 | 0.1756 | 0.777 | 0.848 | 0.267 |
| Chapman | 104729 | chest_only | clean | 0.6522 | 0.9035 | 0.1440 | 0.1830 | 0.753 | 0.882 | 0.235 |
| Chapman | 104729 | chest_only | RA_LL_reversal | 0.6522 | 0.9035 | 0.1440 | 0.1830 | 0.753 | 0.882 | 0.235 |
| Chapman | 130363 | clean_12lead | clean | 0.6513 | 0.9145 | 0.1220 | 0.1201 | 0.772 | 0.883 | 0.237 |
| Chapman | 130363 | clean_12lead | RA_LL_reversal | 0.3403 | 0.7044 | 0.5300 | 0.6101 | 0.870 | 0.338 | 0.700 |
| Chapman | 130363 | physics_aware_12lead | clean | 0.6557 | 0.9143 | 0.1743 | 0.2350 | 0.767 | 0.876 | 0.242 |
| Chapman | 130363 | physics_aware_12lead | RA_LL_reversal | 0.6287 | 0.8949 | 0.2043 | 0.2809 | 0.740 | 0.867 | 0.244 |
| Chapman | 130363 | chest_only | clean | 0.6429 | 0.9013 | 0.1732 | 0.2572 | 0.716 | 0.888 | 0.223 |
| Chapman | 130363 | chest_only | RA_LL_reversal | 0.6429 | 0.9013 | 0.1732 | 0.2572 | 0.716 | 0.888 | 0.223 |
| Chapman | 155921 | clean_12lead | clean | 0.6609 | 0.9160 | 0.1049 | 0.0947 | 0.761 | 0.881 | 0.237 |
| Chapman | 155921 | clean_12lead | RA_LL_reversal | 0.3679 | 0.6792 | 0.4488 | 0.5389 | 0.875 | 0.280 | 0.748 |
| Chapman | 155921 | physics_aware_12lead | clean | 0.6625 | 0.9120 | 0.1159 | 0.1267 | 0.745 | 0.882 | 0.233 |
| Chapman | 155921 | physics_aware_12lead | RA_LL_reversal | 0.6447 | 0.8967 | 0.1365 | 0.1691 | 0.772 | 0.853 | 0.262 |
| Chapman | 155921 | chest_only | clean | 0.6559 | 0.9011 | 0.1226 | 0.1396 | 0.732 | 0.886 | 0.228 |
| Chapman | 155921 | chest_only | RA_LL_reversal | 0.6559 | 0.9011 | 0.1226 | 0.1396 | 0.732 | 0.886 | 0.228 |
| Ningbo | 104729 | clean_12lead | clean | 0.6736 | 0.9219 | 0.0973 | 0.0878 | 0.735 | 0.910 | 0.186 |
| Ningbo | 104729 | clean_12lead | RA_LL_reversal | 0.1773 | 0.5541 | 0.5032 | 0.5602 | 0.744 | 0.336 | 0.676 |
| Ningbo | 104729 | physics_aware_12lead | clean | 0.6955 | 0.9295 | 0.1094 | 0.1353 | 0.743 | 0.909 | 0.188 |
| Ningbo | 104729 | physics_aware_12lead | RA_LL_reversal | 0.6741 | 0.9198 | 0.1135 | 0.1417 | 0.723 | 0.910 | 0.184 |
| Ningbo | 104729 | chest_only | clean | 0.6607 | 0.9105 | 0.1073 | 0.1107 | 0.712 | 0.911 | 0.181 |
| Ningbo | 104729 | chest_only | RA_LL_reversal | 0.6607 | 0.9105 | 0.1073 | 0.1107 | 0.712 | 0.911 | 0.181 |
| Ningbo | 130363 | clean_12lead | clean | 0.6756 | 0.9242 | 0.1089 | 0.1164 | 0.738 | 0.909 | 0.187 |
| Ningbo | 130363 | clean_12lead | RA_LL_reversal | 0.1878 | 0.5781 | 0.5157 | 0.5769 | 0.735 | 0.393 | 0.626 |
| Ningbo | 130363 | physics_aware_12lead | clean | 0.6761 | 0.9200 | 0.0909 | 0.0780 | 0.724 | 0.903 | 0.190 |
| Ningbo | 130363 | physics_aware_12lead | RA_LL_reversal | 0.6647 | 0.9141 | 0.0979 | 0.0917 | 0.735 | 0.891 | 0.201 |
| Ningbo | 130363 | chest_only | clean | 0.6581 | 0.9111 | 0.1121 | 0.1474 | 0.705 | 0.908 | 0.182 |
| Ningbo | 130363 | chest_only | RA_LL_reversal | 0.6581 | 0.9111 | 0.1121 | 0.1474 | 0.705 | 0.908 | 0.182 |
| Ningbo | 155921 | clean_12lead | clean | 0.6791 | 0.9227 | 0.0907 | 0.0892 | 0.725 | 0.912 | 0.182 |
| Ningbo | 155921 | clean_12lead | RA_LL_reversal | 0.1814 | 0.5553 | 0.5162 | 0.5902 | 0.811 | 0.230 | 0.776 |
| Ningbo | 155921 | physics_aware_12lead | clean | 0.6890 | 0.9230 | 0.1228 | 0.1607 | 0.742 | 0.905 | 0.191 |
| Ningbo | 155921 | physics_aware_12lead | RA_LL_reversal | 0.6607 | 0.9093 | 0.1356 | 0.1866 | 0.723 | 0.902 | 0.191 |
| Ningbo | 155921 | chest_only | clean | 0.6474 | 0.9027 | 0.1198 | 0.1433 | 0.710 | 0.905 | 0.186 |
| Ningbo | 155921 | chest_only | RA_LL_reversal | 0.6474 | 0.9027 | 0.1198 | 0.1433 | 0.710 | 0.905 | 0.186 |

Validation thresholds are model- and seed-specific 90th percentiles of negative validation scores. Sensitivity/specificity comparisons therefore use each model's own frozen validation operating point; they do not establish clinical superiority.

## Paired comparisons

Positive AUPRC/AUROC deltas favor the left-hand 12-lead comparator; positive Brier deltas mean its squared probability error is higher. Intervals resample canonical test records synchronously and average the paired seed-level differences.

| Dataset | Endpoint | Comparison | Metric | Mean paired difference | Bootstrap 95% interval |
|---|---|---|---|---:|---:|
| Chapman | TWA | physics_aware_12lead_minus_chest_only_clean | AUPRC | +0.0137 | [-0.0017, +0.0299] |
| Chapman | TWA | physics_aware_12lead_minus_chest_only_clean | AUROC | +0.0119 | [+0.0058, +0.0185] |
| Chapman | TWA | physics_aware_12lead_minus_chest_only_clean | Brier | -0.0107 | [-0.0139, -0.0074] |
| Chapman | TWA | physics_aware_12lead_minus_chest_only_RA_LL | AUPRC | -0.0131 | [-0.0282, +0.0008] |
| Chapman | TWA | physics_aware_12lead_minus_chest_only_RA_LL | AUROC | -0.0074 | [-0.0116, -0.0031] |
| Chapman | TWA | physics_aware_12lead_minus_chest_only_RA_LL | Brier | +0.0153 | [+0.0121, +0.0186] |
| Chapman | TWA | clean_12lead_minus_chest_only_clean | AUPRC | +0.0028 | [-0.0220, +0.0281] |
| Chapman | TWA | clean_12lead_minus_chest_only_clean | AUROC | +0.0136 | [+0.0050, +0.0232] |
| Chapman | TWA | clean_12lead_minus_chest_only_clean | Brier | -0.0314 | [-0.0366, -0.0263] |
| Chapman | TWA | clean_12lead_minus_chest_only_RA_LL | AUPRC | -0.3225 | [-0.3632, -0.2797] |
| Chapman | TWA | clean_12lead_minus_chest_only_RA_LL | AUROC | -0.2405 | [-0.2652, -0.2148] |
| Chapman | TWA | clean_12lead_minus_chest_only_RA_LL | Brier | +0.3743 | [+0.3610, +0.3888] |
| Ningbo | TWA | physics_aware_12lead_minus_chest_only_clean | AUPRC | +0.0315 | [+0.0178, +0.0438] |
| Ningbo | TWA | physics_aware_12lead_minus_chest_only_clean | AUROC | +0.0161 | [+0.0116, +0.0209] |
| Ningbo | TWA | physics_aware_12lead_minus_chest_only_clean | Brier | -0.0054 | [-0.0076, -0.0032] |
| Ningbo | TWA | physics_aware_12lead_minus_chest_only_RA_LL | AUPRC | +0.0111 | [-0.0014, +0.0231] |
| Ningbo | TWA | physics_aware_12lead_minus_chest_only_RA_LL | AUROC | +0.0063 | [+0.0022, +0.0104] |
| Ningbo | TWA | physics_aware_12lead_minus_chest_only_RA_LL | Brier | +0.0026 | [+0.0007, +0.0047] |
| Ningbo | TWA | clean_12lead_minus_chest_only_clean | AUPRC | +0.0207 | [+0.0025, +0.0387] |
| Ningbo | TWA | clean_12lead_minus_chest_only_clean | AUROC | +0.0148 | [+0.0094, +0.0205] |
| Ningbo | TWA | clean_12lead_minus_chest_only_clean | Brier | -0.0141 | [-0.0168, -0.0111] |
| Ningbo | TWA | clean_12lead_minus_chest_only_RA_LL | AUPRC | -0.4732 | [-0.4981, -0.4472] |
| Ningbo | TWA | clean_12lead_minus_chest_only_RA_LL | AUROC | -0.3456 | [-0.3628, -0.3292] |
| Ningbo | TWA | clean_12lead_minus_chest_only_RA_LL | Brier | +0.3986 | [+0.3903, +0.4070] |

## Interpretation limits

If physics-aware 12-lead performance exceeds chest-only, the result supports practical robustness beyond removal of limb leads in these datasets; it does not establish which leads or internal features the network uses. If chest-only is similar or better, it is a strong simpler baseline and changes the practical interpretation of augmentation.
Ningbo is a corrected reanalysis after historical test access, not an untouched independent confirmation. The chest-only models are newly fit but evaluated on that previously accessed test split. Results do not establish clinical utility, equivalence, or general architecture superiority. Calibration and validation-fixed operating points are reported descriptively; no extra calibration or mostly-clean mixture analysis was performed.

## Answers to the prespecified questions

1. **Chest-only versus 12-lead models.** The ordinary 12-lead model loses mean TWA AUPRC under RA-LL by about 0.325 on Chapman and 0.494 on Ningbo, while the chest-only scores are exactly invariant. Relative to the physics-aware model, chest-only clean-condition TWA AUPRC is lower by 0.0137 on Chapman (paired 95% CI -0.0017 to +0.0299) and 0.0315 on Ningbo (+0.0178 to +0.0438). Under RA-LL, physics-aware minus chest-only is -0.0131 on Chapman (-0.0282 to +0.0008) and +0.0111 on Ningbo (-0.0014 to +0.0231). These are corrected reanalyses of previously accessed test sets, especially Ningbo.

2. **Augmentation beyond excluding limb leads.** Physics-aware augmentation improves RA-LL TWA AUPRC over the ordinary 12-lead model by a mean 0.309 on Chapman and 0.484 on Ningbo. Against chest-only under RA-LL, however, the paired AUPRC intervals include zero in both datasets. This supports the practical value of augmentation over a vulnerable 12-lead model, while these results do not show a consistent primary-endpoint advantage over simply using chest leads.

3. **Ningbo.** The large improvement of physics-aware over ordinary 12-lead performance under RA-LL recurs in Ningbo. The additional advantage over chest-only is not established on RA-LL TWA AUPRC. Ningbo was historically accessed; report it as corrected reanalysis rather than an independent untouched confirmation.

4. **Clean performance, calibration, and operating points.** Mean clean TWA AUPRC was not lower for physics-aware than ordinary 12-lead training (descriptive deltas +0.0109 Chapman and +0.0108 Ningbo). Calibration was less stable: mean physics-aware-minus-ordinary Brier deltas were +0.0207 and +0.0087, and ECE-15 deltas +0.0492 and +0.0269, respectively; higher values indicate worse calibration. The per-model, per-seed validation-frozen sensitivity and specificity are shown above. They are operating-point descriptions from different frozen thresholds, not a clinical-superiority test.

## Evaluation correction audit

The first evaluation attempt raised a caller-signature TypeError before writing any prediction. It had read test labels before writing that attempt's evaluation-freeze record; this access is preserved in INITIAL_EVALUATION_INCIDENT.json. All six training units had already completed and were not repeated. The corrected evaluator writes the freeze record before reading test labels or waveforms. The training protocol and its SHA-256 were left unchanged; the evaluator-only correction and source hashes are in EVALUATION_CORRECTION_AMENDMENT_001.json. The final results are from the corrected pass.

Saved clean and RA-LL comparator predictions were replayed against the freshly inferred outputs: all 24/24 matched exactly (maximum absolute difference 0.0; frozen tolerance 1e-5 absolute and 1e-4 relative). Individual prediction hashes were available and verified for 22/24 saved predictions. The two Chapman v4 files without individual prediction hashes were checked against exact checkpoint replay on the bound test order. Details are in SAVED_COMPARATOR_REPLAY_AUDIT.json.

## Reproducibility

The protocol, six model manifests, comparator audit, inference manifests, source hashes, training histories, validation outputs, checkpoint hashes, synchronized bootstrap indices, and paired tables are stored in this directory. Existing source checkpoints, frozen protocols, and historical outputs were not modified.
