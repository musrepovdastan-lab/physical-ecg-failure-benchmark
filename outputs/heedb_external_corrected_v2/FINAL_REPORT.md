# HEEDB I0006 corrected-v2 external evaluation

## Decision

This is a **post-hoc implementation correction**, not a replacement for frozen v1 and not an exact Chapman/Ningbo SNOMED-label replication. With the two documented corrections, clean diagnostic ranking **does transfer** to the 10,000-patient HEEDB physician-code proxy cohort: three-seed core-TWA AUPRC is 0.5770 (baseline) and 0.5715 (physics-aware), versus 0.1524 prevalence; AUROC is 0.8961 and 0.8854. The clean result is not the near-prevalence ranking seen in v1.

Physics-aware training gives a **useful absolute benefit under electrically exact RA–LL**, not merely a smaller drop: core-TWA reversal AUPRC is 0.5664, AUROC 0.8837, sensitivity 0.6649 and specificity 0.8885 at the unchanged Ningbo threshold. Baseline under the same exact reversal has AUPRC 0.1774, AUROC 0.5530, sensitivity 0.7117 and specificity 0.3688. Thus the physics-aware arm greatly improves ranking and specificity but **does not improve frozen-threshold TWA sensitivity**; its sensitivity is 0.0468 lower (paired 95% CI −0.0706 to −0.0255). This operating-point tradeoff must not be obscured by the AUPRC gain.

## What changed from v1

Only the [frozen corrected-v2 plan](../heedb_failure_audit_v1/CORRECTED_V2_PLAN.md)'s two implementation corrections were made: the locally cached 5,000-sample, 500 Hz microvolt waveform is sampled with `::5` to exactly `(12, 1000)` as in Ningbo training; exact RA–LL uses `src.ecg_physics.named_reversal` followed by `derive_display` before the unchanged per-lead Ningbo normalization. The 250 Hz recordings had already been resampled 2-to-1 into that same cached 5,000-sample form. The same patients, labels, six checkpoints and preprocessing files, validation thresholds, 24 masks, and bootstrap specification were retained. No remapping, training, calibration, S3 access or download occurred.

Three synthetic/unit tests required exact sample positions and shape and checked **all 12** RA–LL leads, including corrected aVR and aVF, against the physics library. The pre-inference engineering smoke passed on all 23 excluded ECGs (one original 250 Hz, 22 at 500 Hz), six checkpoints and 26 conditions, with finite three-channel probabilities and exact library reversal. Inference ran once on 10,000 patients; no unreadable replacements were needed. The v2 [protocol](../../configs/FROZEN_HEEDB_EXTERNAL_CORRECTED_V2_PROTOCOL.json) and [evaluator](../../src/heedb_external_corrected_v2.py) were hashed before inference. Protocol SHA-256 is `c339031084061dddf683f263319fd80c9d87bcec47df781eeeef0a3ed3399838`; evaluator SHA-256 is `fca478a2fd0c5ac5def5cdb3e3a3a05cae14546d0e3a82b67c76281cf7812e3d`. The seven bound public v1 artifacts still match their pre-v2 hashes.

## Primary core-TWA comparisons

Three-seed means of seed-level metrics; AP is AUPRC. Positive `clean − RA–LL` indicates an AP reversal penalty. All intervals are percentile 95% intervals from 1,000 synchronized, paired patient bootstrap resamples. These are contrasts of the fixed 10,000 patients, not a new selected population.

| Contrast | AP estimate [95% CI] | Sensitivity estimate [95% CI] |
|---|---:|---:|
| Baseline clean − exact RA–LL | 0.3996 [0.3783, 0.4241] | −0.0956 [−0.1218, −0.0706] |
| Physics-aware clean − exact RA–LL | 0.0052 [−0.0010, 0.0118] | −0.0210 [−0.0302, −0.0115] |
| Physics-aware − baseline, clean | −0.0055 [−0.0162, 0.0041] | 0.0278 [0.0153, 0.0388] |
| Physics-aware − baseline, exact RA–LL | **0.3890 [0.3666, 0.4126]** | **−0.0468 [−0.0706, −0.0255]** |

Under reversal, baseline's predicted-positive fraction rises to 0.6435 and specificity falls to 0.3688, explaining why its sensitivity rises even while ranking collapses. The physics-aware reversed predicted-positive fraction is 0.1958, close to its clean 0.1847; its specificity stays 0.8885. No threshold was selected or changed on HEEDB.

## V1 versus corrected v2

V1's RA–LL condition used incorrect aVR/aVF equations and both v1 conditions fed 5,000 rather than 1,000 samples; **v1 reversal values are not estimates of the intended electrically exact condition**. They remain here for transparent provenance, not as a formal paired v1–v2 effect analysis.

| Core TWA; prevalence 0.1524 | V1 baseline | V1 physics-aware | Corrected v2 baseline | Corrected v2 physics-aware |
|---|---:|---:|---:|---:|
| Clean AUPRC | 0.1579 | 0.1440 | 0.5770 | 0.5715 |
| Clean AUROC | 0.4837 | 0.4525 | 0.8961 | 0.8854 |
| Clean frozen-threshold sensitivity | 0.0809 | 0.0151 | 0.6161 | 0.6439 |
| RA–LL AUPRC | 0.1616 | 0.1493 | 0.1774 | 0.5664 |
| RA–LL AUROC | 0.5093 | 0.4583 | 0.5530 | 0.8837 |
| RA–LL frozen-threshold sensitivity | 0.0225 | 0.0103 | 0.7117 | 0.6649 |

## Other outcomes and sensitivity labels

All labels are from physician-overread `codes_physician` only, with **1,524 core TWA, 1,665 core SB, 673 core STach, 770 strict TWA and 1,448 strict SB** positives among 10,000 patients. Software and v24 codes were never combined. Strict TWA/SB are sensitivity definitions, not replacements for the core primary outcomes.

| Mapping/outcome (prevalence) | Arm | Clean AP / AUROC | Exact RA–LL AP / AUROC | Exact RA–LL sensitivity / specificity |
|---|---|---:|---:|---:|
| Core SB (0.1665) | Baseline | 0.9726 / 0.9946 | 0.8739 / 0.9667 | 0.9612 / 0.8070 |
| Core SB (0.1665) | Physics-aware | 0.9745 / 0.9950 | 0.9737 / 0.9949 | 0.9972 / 0.8456 |
| Core STach (0.0673) | Baseline | 0.9559 / 0.9959 | 0.6084 / 0.9246 | 0.8504 / 0.8164 |
| Core STach (0.0673) | Physics-aware | 0.9569 / 0.9957 | 0.9581 / 0.9956 | 0.9931 / 0.9002 |
| Strict TWA (0.0770) | Baseline | 0.3308 / 0.8789 | 0.1160 / 0.6376 | 0.8156 / 0.3709 |
| Strict TWA (0.0770) | Physics-aware | 0.3334 / 0.8720 | 0.3319 / 0.8714 | 0.7346 / 0.8492 |
| Strict SB (0.1448) | Baseline | 0.8757 / 0.9830 | 0.7452 / 0.9532 | 0.9595 / 0.7872 |
| Strict SB (0.1448) | Physics-aware | 0.8633 / 0.9830 | 0.8690 / 0.9833 | 0.9982 / 0.8244 |

The physics-aware minus baseline exact-RA–LL AP contrast is 0.0999 [0.0883, 0.1119] for core SB, 0.3497 [0.3194, 0.3809] for core STach, and 0.2159 [0.1920, 0.2409] for strict TWA. All three TWA seeds show strong clean ranking and a large exact-reversal AP advantage for physics-aware training; the corrected result is not driven by one seed. Full per-seed metrics, frozen-threshold predicted-positive fractions, calibration and all core/strict intervals are in the CSV artifacts below.

The deterministic **24-mask subset** has core-TWA mean AP 0.5094 baseline and 0.5135 physics-aware; paired subset arm difference 0.0041, bootstrap 95% CI [−0.0039, 0.0120]. This is a subset, **not** the all-792-mask mean, and does not establish a mask-panel benefit.

## Scope, exclusions, and artifacts

V2 reused the v1 label-blind, one-record-per-patient cohort. The upstream v1 exclusions were 27,328 joined rows with missing patient IDs, 675 with empty physician codes after linkage, and 23 readiness-sample patients; 349,482 patients were eligible after deduplication and the same 10,000 were selected. No unreadable-record replacements occurred. HEEDB physician 12SL constructs overlap with but are not identical to the Chapman/Ningbo SNOMED outcomes; core versus strict label differences and ambiguous T-wave codes remain clinical construct limitations. This supports **external transportability under the fixed HEEDB proxy labels**, not exact-label replication or a new clinical deployment claim.

The new restricted prediction array is 18,720,128 bytes in `data/heedb_restricted/I0006/external_corrected_v2/`, which is git-ignored. **New downloaded bytes: 0.** Disk available at verification: 164,956,168,192 bytes (153.627 GiB). Frozen inference elapsed 156.718 seconds; the 1,000-replicate analysis completed at 2026-09-24 16:47:09 UTC. The [provenance](RUN_PROVENANCE.json), [three-seed means](THREE_SEED_MEAN_METRICS.csv), [per-seed aggregates](AGGREGATE_RESULTS_BY_SEED.csv), [contrasts](AGGREGATE_CONTRASTS.csv), [bootstrap intervals](PAIRED_BOOTSTRAP_INTERVALS.csv), [counts](AGGREGATE_LABEL_COUNTS.json), and [progress log](PROGRESS.jsonl) contain aggregate information only. No patient- or recording-level material was placed in these outputs.
