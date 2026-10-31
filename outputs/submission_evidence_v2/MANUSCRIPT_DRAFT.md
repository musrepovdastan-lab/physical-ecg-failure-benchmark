# When limb-cable reversals disrupt ECG classifiers: a matched stress test and comparison of protection strategies

**Working manuscript draft.** Evidence is limited to the frozen datasets, endpoints, checkpoints, transformations and operating points described below. This draft does not claim clinical validation or a novel universal architecture.

## Abstract

**Background:** Missing-lead tests and physical terminal reversals alter an ECG in different ways. Robustness measured under one input corruption may not predict performance under another.

**Objective:** To compare ordinary training, five-lead masking, simulated physical reversal training, their combination, chest-only input, and a published detector/correction pipeline under specified reversal conditions, while documenting cross-dataset and calibration tradeoffs.

**Methods:** We studied three multilabel ECG interpretation endpoints (TWA primary; sinus bradycardia and sinus tachycardia secondary) in Chapman and Ningbo and evaluated frozen Ningbo models on physician-code proxy labels in HEEDB I0006 (Emory). We corrected an inverted five-lead masking evaluator and repeated affected mask analyses using retained checkpoints. The exact RA–LL mask suppressed I, II, III, aVR, and aVF; the physical RA–LL transform was independently defined by terminal algebra. In the Ningbo four-arm comparison, training mixtures were clean (100% clean), masking (50% clean/50% five-lead masks), physics-aware (50% clean/50% reversal), and combined (50% clean/25% mask/25% reversal). A later external extension compared six frozen pipelines over clean/reversal mixtures with source-validation calibration and validation-B operating thresholds.

**Results:** In corrected Ningbo TWA results, ordinary and physics-aware ResNet seed-mean AUPRC were 0.1822 and 0.6665 under RA–LL (difference +0.4843), versus 0.6761 and 0.6869 on clean input. The recovery report's paired-bootstrap mean difference was +0.4836 (95% CI +0.4564 to +0.5090). The corrected full-sample mean-mask-minus-RA–LL contrast reduction was +0.4657 (95% CI +0.4400 to +0.4891). The exact-mask and 792-mask comparisons remain distinct from physical reversal. Chest-only outputs were invariant to simulated limb reversals; physics-aware minus chest-only RA–LL TWA AUPRC was −0.0131 (95% CI −0.0282 to 0.0008) on Chapman and +0.0111 (−0.0014 to +0.0231) on Ningbo. TimePatchTransformer improved under RA–LL but failed the frozen M3 retention gate because one seed's mean-mask TWA AUPRC decline was 0.1126 (maximum allowed 0.04); Ningbo M3 was not performed. On HEEDB, combined-model TWA AUPRC was 0.592 at q=0 and 0.554 at q=1, versus 0.576 and 0.568 for physics-aware ResNet and 0.572 and 0.581 for the Huang pipeline. The combined-versus-physics-aware clean direction differed between Ningbo and HEEDB. Source-validation calibration reduced Brier/ECE while paired threshold transformation preserved decisions in the checked audit.

**Conclusions:** Physics-aware training substantially improved the tested ResNet's ranking under simulated RA–LL relative to ordinary training. Corrected mask controls, chest-only results, architecture retention, calibration and external mixtures show that conclusions depend on the condition and comparator. The evidence supports a bounded evaluation and comparative tradeoff study, not clinical benefit, a universal winner, or validation against real miswirings.

## Introduction

Automated ECG systems rely on the lead configuration presented to the classifier. A missing displayed lead and a physical terminal swap are different interventions: missingness removes a signal, whereas a terminal reversal changes a defined set of derived leads while leaving others invariant. Performance under one condition should therefore not be treated as a proxy for the other without direct evaluation.

Prior work already studies ECG lead reversal. Cao, Shi and Wu (ICASSP 2020), “Reversal No Longer Matters: Attention-Based Arrhythmia Detection with Lead-Reversal ECG Data,” DOI [10.1109/ICASSP40776.2020.9053307](https://doi.org/10.1109/ICASSP40776.2020.9053307), precludes broad claims that reversal-robust ECG modeling or simulated reversal evaluation began here. The locally reviewed primary-text access was incomplete; its accessible methods support ordinary training with reversal test sets. We do not attribute reversal augmentation to Cao without a complete primary-text verification. The present work is positioned as a comparison of a particular prespecified reversal family, corrected mask controls, training strategies, chest-only input, selected architectures and a detector/correction chain. The contribution is the bounded comparison and the tradeoffs it reveals.

We ask: (1) how does ordinary training perform under a physically specified RA–LL swap compared with reversal-aware training? (2) how do missing-lead controls, chest-only input, and alternative architectures change interpretation? (3) how do four training mixtures compare over mostly-clean and high-fault evaluation mixtures? (4) what does source-validation calibration change about probability error and operating decisions?

## Methods

### Datasets and endpoints

The source test datasets are Chapman (2,050 ECGs) and Ningbo (6,981 ECGs). Each model predicts TWA (primary), SB and STach (secondary), using fixed SNOMED-coded labels in the source protocols: TWA `164934002`, SB `426177001`, and STach `427084000`. For external HEEDB I0006 (Emory), physician-code mappings are proxies and are not asserted to be equivalent to the SNOMED labels. The initial corrected external analysis and the later patient-disjoint confirmation used two different 10,000-patient cohorts from Emory, each with one ECG per patient. The six-method extension reused the latter confirmation cohort after prior access to its outcomes; it did not evaluate a third cohort or a new external center. Clinical review of endpoint mappings remains outstanding.

### Physical transformation and missing-lead controls

The canonical lead order is I, II, III, aVR, aVL, aVF, V1–V6. Exact RA–LL physical terminal exchange changes derived leads I, II, III, aVR and aVF; aVL and V1–V6 remain invariant. The exact five-lead mask removes indices `[0,1,2,3,5]`, namely I, II, III, aVR and aVF, replacing them with the frozen training-mean sentinel before normalization while leaving the remaining seven channels unchanged. The mask shares the affected lead names with the swap but does not reproduce its waveform transformation. The complete mask panel contains all 792 combinations of five displayed leads; metrics are computed separately by mask and then summarized, without averaging prediction vectors.

An audit found that earlier Ningbo mask code retained the selected five leads and replaced the other seven, the opposite of the intended operation. Historical packages and their bound sources were retained. A corrected evaluator was used for the versioned recovery analysis; the previous exact-mask Ningbo AUPRC near 0.527 and conclusions based on the inverted panel are superseded.

### Models and training comparisons

The ResNet four-arm Ningbo follow-up used three seeds. The four arms were: clean, 100% clean training examples; masking, 50% clean and 50% five-lead masks; physics-aware, 50% clean and 50% physical reversals; and combined, 50% clean, 25% masks and 25% physical reversals. Masking and combined were newly fitted; clean and physics-aware checkpoints were reused after binding/integrity checks. This is a follow-up after prior test access, not untouched confirmation. Combined receives half the physical-reversal exposure of physics-aware, so their comparison cannot isolate the causal effect of adding masking or select an optimal augmentation probability `p_aug`.

Chest-only ResNets use V1–V6 with train-derived per-channel normalization and do not receive limb leads, masking or reversal augmentation. TimePatchTransformer M3 compared clean and physics-aware training on Chapman under a frozen retention gate. ECGFounder was evaluated only as a frozen pretrained backbone with a trained head; it was not tested for mitigation.

### Calibration, operating points and mixtures

Platt calibration was fit on clean source-validation subset A; raw operating thresholds were the 90th percentile of negative scores on subset B. The split contained 1,744 records in each subset. Historical checkpoint selection had already used the full validation set before this A/B split, so calibration is not independent of checkpoint selection. Each evaluated pipeline used its frozen validation-B threshold; the Huang primary operating point is the threshold for the complete detector → predicted correction → clean ResNet pipeline. A historical HEEDB Huang threshold is reported only in a separate sensitivity analysis. Raw ranking scores and raw Brier/ECE are distinguished from calibrated Brier/ECE and corresponding threshold-based sensitivity, specificity and predicted-positive fraction.

The external extension forms weighted pooled mixtures of each patient's clean and reversed prediction copies with weights `(1−q_fault)` and `q_fault`. Evaluated q values were 0, 0.001, 0.005, 0.01, 0.05, 0.1 and 1. These are constructed scenarios and are not observed fault prevalence. `q_fault` is distinct from the training augmentation probability `p_aug`. Patient bootstrap intervals use synchronized multiplicities and are conditional on selected checkpoints, calibrators and thresholds; they do not capture all training uncertainty. Three seeds provide limited coverage of that uncertainty.

### Analysis and implementation runtime

Primary evaluation uses TWA AUPRC. AUROC, Brier, ECE-15 and validation-threshold operating metrics provide complementary descriptions; SB and STach are secondary. Paired comparisons use the frozen patient/record bootstrap artifacts; no new inference or bootstrap was run for this synthesis. Runtime values are from the frozen integrated deployment v4 implementation, measured by direct end-to-end CPU request/response on a balanced 128-record Ningbo fixture, one framework thread and fixed CPU affinity over seven runs.

## Results

### Corrected reversal and mask results

For Ningbo TWA, ordinary ResNet seed-mean AUPRC was 0.6761 on clean ECG and 0.1822 under RA–LL; physics-aware ResNet was 0.6869 and 0.6665, respectively. The full-sample seed-mean RA–LL difference is +0.4843; the report's paired-bootstrap mean difference is +0.4836 (95% interval +0.4564 to +0.5090). The corrected full-sample mean-mask-minus-RA–LL contrast reduction was +0.4657 (95% paired interval +0.4400 to +0.4891). Chapman RA–LL TWA AUPRC increased from 0.3278 to 0.6372 (+0.3094) for the corresponding ordinary-to-physics-aware comparison. These are corrected analyses of specified simulations, not observations of real acquisition failures. See `../mask_semantics_recovery_v2/FINAL_RECOVERY_REPORT.md` and `PRIMARY_CONTRASTS.csv`.

On Ningbo, masking minus clean-trained TWA AUPRC was −0.0130 under RA–LL (95% CI −0.0192 to −0.0060) and +0.0598 across the 792-mask mean (+0.0502 to +0.0696). Combined minus physics-aware was −0.0101 on clean (−0.0196 to −0.0009), −0.0218 under RA–LL (−0.0325 to −0.0122) and +0.0371 across the mask mean (+0.0288 to +0.0446). This profile must be read with the unequal reversal exposure and the three-seed limitation. See `../ningbo_four_arm_replication_v1/FINAL_FOUR_ARM_REPORT.md`.

The corrected Ningbo mask controls retain the distinction between masking the same affected lead names and averaging across all five-lead removals:

| Training arm | Clean TWA AUPRC | RA–LL | Mean of 792 masks | Exact affected-lead mask |
|---|---:|---:|---:|---:|
| Ordinary | 0.6761 | 0.1822 | 0.5863 | 0.4358 |
| Physics-aware | 0.6869 | 0.6665 | 0.6049 | 0.6260 |

Values are full-cohort means of three seed-specific metrics from `../mask_semantics_recovery_v2/PRIMARY_CONTRASTS.csv`. In the ordinary arm, the mean-mask-minus-RA–LL difference is +0.4042 (95% CI +0.3809 to +0.4257), and the exact-mask-minus-RA–LL difference is +0.2536 (+0.2316 to +0.2778). The 792-mask average summarizes a specified panel; it does not imply that every individual mask is less damaging than RA–LL.

### Chest-only control and architecture results

Chest-only TWA predictions were exactly invariant between clean and simulated RA–LL input in the checked inference path. Physics-aware minus chest-only RA–LL AUPRC was −0.0131 (95% CI −0.0282 to +0.0008) in Chapman and +0.0111 (−0.0014 to +0.0231) in Ningbo. On clean ECG, the differences were +0.0137 (−0.0017 to +0.0299) and +0.0315 (+0.0178 to +0.0438), respectively. These intervals do not establish equivalence and do not show universal physics-aware superiority. See `../chest_only_baseline_v1/PAIRED_CONTRASTS.csv`.

TimePatchTransformer showed an RA–LL TWA improvement, but M3A failed its frozen architecture-specific criterion: for seed 155921, physics-aware mean-mask AUPRC declined by 0.1126, exceeding the maximum permitted 0.04 decline. Other listed M3A conditions, including all three RA–LL gains, passed. This is a mask-retention failure rather than absence of the measured RA–LL effect. No Stage B or Ningbo Transformer analysis followed. ECGFounder results concern a frozen backbone with trained heads only; no Founder mitigation was tested. See `../architecture_replication_m3_recovery_v2/FINAL_M3_DECISION.md` and `../ecgfounder_baseline_v2/`.

### External six-method comparison and calibration

The corrected TWA HEEDB table is in `EXTERNAL_SIX_METHOD_CORRECTED_SUMMARY.md` and has one row per method/q. At q=0, raw AUPRC was 0.592 for combined, 0.576 physics-aware, 0.572 Huang, 0.568 masking, 0.573 ordinary and 0.537 chest-only. At q=0.1, the corresponding values were 0.587, 0.575, 0.573, 0.418, 0.421 and 0.537. At q=1, values were 0.554, 0.568, 0.581, 0.159, 0.165 and 0.537. Combined exceeded Huang at low q but not q=1; the interval for combined-minus-Huang at q=1 excludes zero in the unfavorable direction. Combined's clean q=0 direction versus physics-aware was positive on HEEDB but negative on Ningbo; only its high-fault disadvantage has the same direction. Do not describe these scenario rankings as a universal winner.

At q=0.01, calibrated TWA sensitivity/specificity were 0.661/0.910 for combined and 0.617/0.919 for Huang; predicted-positive fractions are in the corrected table and paired sensitivity/specificity intervals are in `../heedb_four_arm_extension_v1/PAIRED_CONTRASTS_BOOTSTRAP.csv`. Source-validation calibration improved measured probability-error metrics relative to raw scores, while a matching monotone transformation of scores and thresholds preserved decisions in the audit. All 108 audit rows had zero operating-decision mismatches; clipping can introduce ties. Thus improved Brier does not itself imply that more cases are called positive. At q=1, masking raw Brier was 0.567 versus calibrated Brier 0.304; its AUPRC was 0.159 and sensitivity 0.693. Calibration did not recover its lost ranking. Improved source-calibrated error on these external proxy labels does not establish valid clinical probability calibration for the intended ECG construct.

### Runtime

On the frozen CPU fixture, the physics-aware worker's warm batch-one median/p95 request-to-response time was 10.59/12.42 ms versus 31.51/34.16 ms for the complete Huang detection, predicted correction and clean ResNet chain (2.97× median ratio). Maximum sampled process-tree working set was 612.2 versus 953.6 MiB; time to ready was 5.82 versus 10.98 seconds. Output agreement was checked before timing. These values apply only to these implementations, the documented AMD Ryzen AI 7 350 laptop, single-thread/affinity configuration, and workload; they are not Huang-paper latencies or clinical workflow evidence. See `../integrated_deployment_v4/FINAL_DECISION.md`.

## Discussion

The corrected analyses support a narrow observation: for the evaluated ResNet configurations, physical reversal-aware training markedly improved TWA ranking under simulated RA–LL relative to ordinary clean training. That does not imply one mitigation dominates all alternatives. Chest-only is invariant by construction, while the paired primary-endpoint intervals do not establish a consistent augmented-model advantage over it. Masking improves panel-average performance in the Ningbo four-arm analysis but does not protect against RA–LL. Combined training exchanges performance across clean, reversal and mask conditions, and its reversal exposure differs from the physics-aware arm. External mostly-clean scenarios show why both the fault mixture weight and operating point matter.

This work does not assert that missing-lead testing is generally inadequate: it shows that a specific physical transformation and specified masking transformations yield different measurements in these experiments. The exact mask is an informative channel-matched control, not an equivalent corruption. The audit itself is a material part of the evidence: earlier results that used the inverse mask should not be silently carried forward.

The external comparison uses proxy labels and historical model access. A post-hoc Huang pipeline with a validation-selected threshold is a comparator, not a reproduction of the original published study or proof of equivalence. The two HEEDB samples share Emory. Calibration was fit on source validation, not on HEEDB test outcomes. Bootstrap intervals are conditional on chosen checkpoints/calibrators, and three seeds incompletely quantify training variation. Crossing zero is not evidence of equivalence.

## Limitations and next decisions

All lead faults are simulated; no held-out reversal composition or real registered miswiring was evaluated. The endpoints are coded ECG interpretation constructs, and a clinician has not confirmed their mapping or the HEEDB proxy relationship. No disease diagnosis, clinical benefit, workflow improvement or prospective safety has been established. Ningbo results are corrected reanalyses after earlier test access. M3 did not pass its global frozen retention criterion and was not replicated on Ningbo. Founder protection was not tested. The literature review of Cao lacks a locally retained complete primary-text copy, so the precise relationship to every listed implementation detail remains for confirmation. Only the stated CPU timing setup was measured.

Before submission, authors should confirm the clinical target and mapping, settle main-text versus supplement placement of the controls and exploratory analyses, and complete primary-text/related-work verification. Additional multicenter clinical evaluation, real miswiring validation and prospectively defined operating thresholds may strengthen later work but are not evidence already obtained.

## Artifact references

Primary corrected/reviewed sources and data modes are indexed in [`SOURCE_OF_TRUTH.md`](SOURCE_OF_TRUTH.md). The independently reviewed HEEDB six-method CSV reconstruction, including paired intervals and separate historical Huang threshold, is [`EXTERNAL_SIX_METHOD_CORRECTED_SUMMARY.md`](EXTERNAL_SIX_METHOD_CORRECTED_SUMMARY.md). Claim boundaries are recorded in [`CLAIMS_AND_EVIDENCE.md`](CLAIMS_AND_EVIDENCE.md).
