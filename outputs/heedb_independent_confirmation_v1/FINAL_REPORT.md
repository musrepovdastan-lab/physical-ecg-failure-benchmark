# HEEDB I0006 independent confirmation

## Decision

The **new, patient-disjoint 10,000-ECG confirmation** reproduces corrected v2's main finding under the frozen HEEDB physician-overread proxy labels. For core T-wave abnormality (TWA), physics-aware minus baseline AUPRC under the **simulated, electrically exact RA–LL swap** is **+0.4036** (1,000 paired patient-bootstrap resamples; 95% CI **0.3785 to 0.4255**). All three matched seeds favor physics-aware training. The absolute physics-aware reversal AUPRC is **0.5683**, versus 0.1647 for baseline and 0.1456 class prevalence; its AUROC is 0.8901. Clean TWA ranking also transports (AUPRC 0.5726 baseline, 0.5760 physics-aware; AUROC 0.9031 and 0.8927).

This is an absolute ranking/specificity benefit, **not a frozen-threshold sensitivity benefit**. Under RA–LL, physics-aware sensitivity is 0.6866 versus baseline 0.6969 (difference −0.0103, 95% CI −0.0355 to 0.0113); specificity is 0.8908 versus 0.3599 (difference +0.5309, 95% CI 0.5211 to 0.5403). The physics-aware arm flags 19.33% of recordings versus baseline's 64.83%. Neither model was recalibrated or threshold-tuned on HEEDB.

These results confirm transportability of ranking under the **fixed HEEDB 12SL physician-code constructs** and a **software-simulated cable reversal**. They do not establish performance on verified real-world wiring errors, exact Chapman/Ningbo SNOMED-label replication, or clinical deployment.

## Primary core-TWA metrics

One recording per patient; three-seed means of seed-level metrics. Sensitivity, specificity and predicted-positive fraction use each seed's unchanged Ningbo validation threshold. ECE has 15 fixed bins; Brier and mean score are in the aggregate CSV.

| Arm | Condition | AUPRC | AUROC | Sensitivity | Specificity | Predicted + | ECE15 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Clean | 0.5726 | 0.9031 | 0.6362 | 0.9140 | 0.1661 | 0.0768 |
| Physics-aware | Clean | 0.5760 | 0.8927 | 0.6646 | 0.8993 | 0.1828 | 0.1120 |
| Baseline | Exact RA–LL | 0.1647 | 0.5383 | 0.6969 | 0.3599 | 0.6483 | 0.5477 |
| Physics-aware | Exact RA–LL | 0.5683 | 0.8901 | 0.6866 | 0.8908 | 0.1933 | 0.1253 |

The baseline's increased reversal sensitivity is accompanied by a very high predicted-positive fraction and low specificity, not preserved discrimination. Physics-aware clean-minus-reversal AUPRC is 0.0077 (95% CI 0.0000 to 0.0154), whereas baseline's is 0.4079 (0.3842 to 0.4304). The clean arm difference in AUPRC is 0.0034 (−0.0081 to 0.0148). Core-TWA prevalence is **1,456/10,000 = 0.1456**.

| Arm | Seed | Clean AUPRC | RA–LL AUPRC | RA–LL sensitivity | RA–LL specificity |
|---|---:|---:|---:|---:|---:|
| Baseline | 104729 | 0.5587 | 0.1513 | 0.6195 | 0.4150 |
| Baseline | 130363 | 0.5801 | 0.1829 | 0.7088 | 0.4187 |
| Baseline | 155921 | 0.5792 | 0.1599 | 0.7624 | 0.2461 |
| Physics-aware | 104729 | 0.5872 | 0.6040 | 0.6951 | 0.9044 |
| Physics-aware | 130363 | 0.5751 | 0.5547 | 0.6944 | 0.8749 |
| Physics-aware | 155921 | 0.5657 | 0.5462 | 0.6703 | 0.8931 |

Seed-level RA–LL AUPRC gains are +0.4527, +0.3718 and +0.3863, respectively. Full seed-level AUROC, calibration, operating points and the paired intervals for AUPRC, sensitivity and specificity are in the linked artifacts below.

## Secondary outcomes and label definitions

All outcomes use `diagnoses_acquisition.csv:codes_physician` only. Core TWA codes are 1140/1145/1150/1151/1160/1161/1170/1171/1172/1180/1181; core sinus bradycardia (SB) 21/24; sinus tachycardia (STach) 23. Strict sensitivity definitions are TWA 1140 and SB 21. Neither software codes nor v24 codes were merged. Mixed/suggestive/serial T-wave codes outside the prespecified core set remain negatives; these construct ambiguities limit direct comparisons with Chapman/Ningbo SNOMED outcomes.

| Outcome (positive/10,000) | Arm | Clean AUPRC | RA–LL AUPRC | RA–LL sensitivity | RA–LL specificity |
|---|---|---:|---:|---:|---:|
| Core SB (1,637) | Baseline | 0.9765 | 0.8853 | 0.9676 | 0.8096 |
| Core SB (1,637) | Physics-aware | 0.9751 | 0.9757 | 0.9971 | 0.8526 |
| Core STach (615) | Baseline | 0.9520 | 0.5953 | 0.8650 | 0.8143 |
| Core STach (615) | Physics-aware | 0.9502 | 0.9485 | 0.9962 | 0.9005 |
| Strict TWA (727) | Baseline | 0.3179 | 0.1004 | 0.7905 | 0.3628 |
| Strict TWA (727) | Physics-aware | 0.3209 | 0.3156 | 0.7565 | 0.8509 |
| Strict SB (1,453) | Baseline | 0.8838 | 0.7773 | 0.9649 | 0.7924 |
| Strict SB (1,453) | Physics-aware | 0.8700 | 0.8742 | 0.9975 | 0.8344 |

Physics-aware minus baseline RA–LL AUPRC is +0.0904 [0.0781, 0.1021] for core SB, +0.3532 [0.3208, 0.3886] for core STach, +0.2152 [0.1912, 0.2401] for strict TWA, and +0.0969 [0.0788, 0.1137] for strict SB. The **strict-TWA sensitivity tradeoff is unfavorable**: −0.0339 [−0.0662, −0.0023], despite its large AUPRC and specificity advantage. These are secondary/sensitivity analyses, not replacement primary definitions. All three seeds for every outcome are in [per-seed aggregates](AGGREGATE_RESULTS_BY_SEED.csv).

## Independent comparison with corrected v2

The two cohorts are disjoint and **not pooled**. Both used the same fixed models, labels and electrically exact transform. This side-by-side comparison is descriptive; the paired confidence interval above is within the new cohort, not a cross-cohort significance test.

| Core TWA | Corrected v2, original 10,000 | Independent confirmation, new 10,000 |
|---|---:|---:|
| Prevalence | 0.1524 | 0.1456 |
| Baseline clean AUPRC | 0.5770 | 0.5726 |
| Physics-aware clean AUPRC | 0.5715 | 0.5760 |
| Baseline exact RA–LL AUPRC | 0.1774 | 0.1647 |
| Physics-aware exact RA–LL AUPRC | 0.5664 | 0.5683 |
| RA–LL physics-aware minus baseline AUPRC | +0.3890 | +0.4036 |
| RA–LL physics-aware minus baseline sensitivity | −0.0468 | −0.0103 |
| RA–LL physics-aware minus baseline specificity | +0.5197 | +0.5309 |

Thus the independent sample supports the corrected-v2 ranking and specificity interpretation, while the size of the frozen-threshold sensitivity disadvantage varies between samples. It does **not** justify saying physics-aware training improved core-TWA sensitivity.

## Exclusions, engineering recovery and reproducibility

The upstream v1 label-blind source had 998,844 joined recordings, 27,328 missing patient IDs and 675 empty physician-code fields after linkage. It contained 349,482 one-record-per-patient candidates **after excluding the 23 readiness patients**. Removing the 10,000 prior v1/corrected-v2 patients left 339,482; a newly salted, label-blind patient hash selected exactly 10,000. No waveform was unreadable and no patient or recording was replaced. The restricted selection hash is `7af9986691871d57df5699e270b3cf4442bf5a733536356155a87094bac09de4`.

The original downloader stopped after 19,999 objects. The [separately hashed engineering amendment](ENGINEERING_AMENDMENT_v1a.md) left the frozen scientific evaluator and protocol intact, resumed only the remaining 120,000-byte `.dat`, and verified all 20,000 objects against exact remote sizes before marking download complete. The original path rejection was not reproducible and is **not** claimed as a proven Windows race. New transfer for the confirmation totaled **1,177,012,861 bytes (1.096 GiB)**, below 2 GiB; the amendment itself transferred only 120,000 bytes. No further S3 access was needed for inference or analysis. The estimated ordinary S3 service charge before discounts/free tier was $0.1219, not a bill. The locally retained waveforms, manifests, cache and predictions are under git-ignored `data/heedb_restricted/I0006/independent_confirmation_v1/`; no patient- or recording-level data are in this output directory.

The scientific [protocol](../../configs/FROZEN_HEEDB_INDEPENDENT_CONFIRMATION_v1.json) SHA-256 is `199344e9f7cbb0eb55564f52ce320bca4c85478ceb68174a3d11568611a58760`; [evaluator](../../src/heedb_independent_confirmation_v1.py) SHA-256 is `c4902b65b8417d6f62c67b3d4286ab9ad09f8eff6097b5cb25e9d5535c67d5b5`. The amendment protocol SHA-256 is `fd001d31bca91bba76f6e197b97e6bee89d9153d101fc9717cb3b3911b13c69a`. The 23 excluded ECGs passed exact `(12,1000)` and all-12-lead physics checks across the six models and two conditions. The single frozen inference took 41.172 seconds and was followed by 1,000 synchronized patient-bootstrap resamples. Artifact hashes, transfer and cache verification, thresholds and run details are in [provenance](RUN_PROVENANCE.json), [three-seed means](THREE_SEED_MEAN_METRICS.csv), [contrasts](AGGREGATE_CONTRASTS.csv), [bootstrap intervals](PAIRED_BOOTSTRAP_INTERVALS.csv), [counts](AGGREGATE_LABEL_COUNTS.json) and the aggregate [progress log](PROGRESS.jsonl).
