# Electrically exact simulated limb-cable reversals expose a diagnostic robustness gap missed by missing-lead tests

## Abstract

ECG classifiers are commonly stress-tested by omitting leads, yet an electrode-cable reversal creates coherent but incorrect displayed waveforms. We compared a clean-trained 12-lead ResNet with matched physics-aware training under electrically exact **simulated** limb-terminal reversals, without altering diagnostic thresholds at evaluation. On Chapman (2,050 test ECGs) and Ningbo (6,981), clean-trained T-wave-abnormality (TWA) AUPRC under RA-LL reversal fell to 0.328 and 0.182, respectively, below its mean across all 792 consequence-count-matched five-lead masks (0.574 and 0.514). Physics-aware RA-LL AUPRC was 0.637 and 0.667. On a new, patient-disjoint 10,000-ECG HEEDB/Emory confirmation using physician-overread **proxy** labels, RA-LL AUPRC was 0.165 unprotected versus 0.568 physics-aware (paired difference +0.404; 95% patient-bootstrap interval 0.379–0.426); clean AUPRC was 0.573 versus 0.576. The large reversal gain was chiefly in ranking and specificity (0.360 versus 0.891), not frozen-threshold sensitivity (0.697 versus 0.687). A published Huang detector followed by predicted correction and the clean ResNet was a strong **post-hoc exploratory** comparator on the same HEEDB cohort: RA-LL AUPRC 0.581; physics-aware minus Huang -0.013 [-0.028, +0.002]. This interval does not establish equivalence or an AUPRC disadvantage. Our bounded result concerns simulations, fixed models and proxy outcomes—not verified clinical wiring errors or patient benefit.

## 1. Introduction

Twelve displayed ECG leads arise from a smaller set of electrode potentials; reversing cables changes a coherent group of leads according to their electrical derivation. A model that tolerates absent channels need not tolerate a plausible, complete, but miswired tracing. We isolate that distinction with an exact terminal permutation and a consequence-count-matched masking panel, then test whether training with the same physical family protects diagnostic ranking. The diagnostic endpoints are TWA, sinus bradycardia (SB) and sinus tachycardia (STach). We focus on TWA; SB/STach and stricter HEEDB definitions are secondary.

### Related work and the specific gap

| Original paper | Contribution | Relation to this study; boundary |
|---|---|---|
| [Zheng et al., *Scientific Data* 2020](https://doi.org/10.1038/s41597-020-0386-x) | Introduced the Chapman–Shaoxing 12-lead arrhythmia research database. | One source of ECGs; it did not establish robustness of our classifiers to exact cable reversal. |
| [Oh et al., CHIL/PMLR 2022](https://proceedings.mlr.press/v174/oh22a.html) | Lead-agnostic self-supervised representations with random lead masking for variable available leads. | Motivates missing-lead stress testing, but masking is not an electrical reversal. Our 792-mask panel is our own matched control, not a reproduction of their model. |
| [Huang et al., *Physiological Measurement* 2024](https://doi.org/10.1088/1361-6579/ad43ae) | Published lightweight detectors for simulated limb/chest lead misplacement, with external detector testing. | Its unmodified limb detector supplies a strong operational detect→predicted-correct→clean-classifier comparator. The paper did not establish our downstream diagnostic AUPRC, and our HEEDB comparison is post-hoc. |
| [Koscova et al., *Scientific Data* 2026](https://doi.org/10.1038/s41597-026-06861-9) | Describes HEEDB, including multi-institution clinical ECGs and 12SL annotations. | Supplies a distinct Emory transportability setting. Our physician-code constructs are proxies, not exact Chapman/Ningbo labels. |

## 2. Methods

### Data, outcomes and frozen comparisons

The Chapman and Ningbo test sets contained 2,050 and 6,981 recordings (377 and 1,034 TWA positives). Their outcome mapping used the project's SNOMED-label definitions, including TWA 164934002, SB 426177001 and STach 427084000. The historic Chapman baseline stress test preceded Chapman physics-aware M1; the Ningbo M2 follow-up was locked after inspecting the earlier Ningbo baseline. We do not present those phases as a single untouched preregistration.

HEEDB v5.0 I0006/Emory evaluation used one ECG per patient selected by a frozen, label-blind hash rule. The independent-confirmation cohort had 10,000 patients and no overlap with the original 10,000-patient v1/corrected-v2 cohort or 23 readiness patients. V1 had implementation faults; corrected v2 repaired input length and reversal implementation on the *same* first 10,000, then the disjoint confirmation ran with fixed criteria. Only the independent cohort supplies the confirmatory HEEDB numbers here; cohorts are not pooled. HEEDB outcomes came exclusively from `diagnoses_acquisition.csv:codes_physician`, not contemporaneous software codes or batch-reprocessed v24 codes. Core TWA used codes 1140/1145/1150/1151/1160/1161/1170/1171/1172/1180/1181 (1,456/10,000); core SB 21/24 (1,637); core STach 23 (615). Strict TWA 1140 (727) and strict SB 21 (1,453) were sensitivity definitions. These code sets contain clinical ambiguity and are **not** silently equated to the SNOMED constructs. Physician review is pending.

### Models and interventions

Three matched-seed 12-lead ResNet clean baselines were compared with three matched-seed physics-aware checkpoints. Physics-aware augmentation used 50% clean inputs and 50% exact RA-LA, RA-LL or LA-LL terminal reversals sampled uniformly, applied before training normalization. Saved models, Ningbo validation thresholds and preprocessing were fixed for external evaluation; there was no HEEDB training, calibration or threshold tuning. HEEDB waveforms were aligned to canonical I, II, III, aVR, aVL, aVF, V1–V6, converted from mV to µV, resampled from 250 to 500 Hz when needed, and sampled at `::5` to exactly 12 × 1,000 steps before checkpoint-specific Ningbo normalization. The tested physics library performed terminal reversal and displayed-lead derivation. Exact simulated RA-LL changes displayed I, II, III, aVR and aVF; aVL and V1–V6 remain invariant. The Chapman/Ningbo control enumerated all C(12,5)=792 displayed five-lead masks. This matches the **number** of changed leads, not the waveform information, and a mask is not a physical cable fault.

For the exploratory HEEDB comparator, Huang's published, unmodified detector received its documented three-lead III/aVR/V6 adapter after 500-Hz filtering, resampling to 120 Hz and within-lead z-scoring. Its argmax prediction—not the known simulated fault—selected whether and how to invert the displayed ECG. Corrected waveforms entered the same saved clean ResNets and unchanged Ningbo thresholds. The Huang comparison was planned *after* the independent HEEDB results were viewed and cannot be promoted to confirmatory evidence.

### Analysis

Primary HEEDB contrast: three-seed mean core-TWA AUPRC under simulated RA-LL, physics-aware minus unprotected. We also report clean ranking, AUROC, fixed-threshold sensitivity, specificity, predicted-positive fraction, Brier score and 15-bin expected calibration error (ECE15). Chapman/Ningbo primary stress contrast was mean five-mask AUPRC minus RA-LL AUPRC, with its reduction after physics-aware training. Each cohort used 1,000 synchronized paired bootstrap replicates: record-level for Chapman/Ningbo saved analyses and patient-level for HEEDB, with percentile 95% intervals. These intervals are descriptive, not multiplicity-adjusted. A CI containing zero is neither equivalence nor noninferiority. The Huang comparison reused the same HEEDB independent patients and bootstrap pairing but was post-hoc exploratory.

## 3. Results

### Missing-lead masking versus exact simulated reversal

| Dataset / arm | Clean TWA AUPRC | Mean of 792 five-lead masks | Simulated RA-LL AUPRC |
|---|---:|---:|---:|
| Chapman, clean-trained | 0.6531 | 0.5738 | 0.3278 |
| Chapman, physics-aware | 0.6641 | 0.5679 | 0.6372 |
| Ningbo, clean-trained | 0.6761 | 0.5139 | 0.1822 |
| Ningbo, physics-aware | 0.6869 | 0.5375 | 0.6665 |

The clean-trained mask-minus-RA-LL AUPRC gap was 0.2457 [0.2096, 0.2796] on Chapman and 0.3311 [0.3113, 0.3503] on Ningbo. Physics-aware training reduced that *matched contrast* by 0.3145 [0.2771, 0.3508] and 0.4598 [0.4348, 0.4839], respectively. Mean values summarize three seeds; intervals are paired bootstraps over the saved test cohorts. SB/STach controls were measured in the frozen analyses but do not establish a universal mechanism.

![Figure 1: Chapman/Ningbo ranking](figures/figure1_cross_dataset_ranking.png)

*Figure 1.* TWA AUPRC across clean ECGs, all 792 five-displayed-lead masks (arithmetic mean), and exact simulated RA-LL. Faint marks are seeds; strong marks are seed means. Dotted horizontal lines mark dataset-specific test prevalence. The masks are consequence-count matched, not waveform-equivalent.

### Patient-disjoint HEEDB confirmation and operating-point tradeoffs

Core-TWA prevalence was 0.1456. The independent HEEDB physics-aware-minus-unprotected RA-LL AUPRC difference was **+0.4036 [0.3785, 0.4255]**, with every matched seed favoring physics-aware ranking. Clean AUPRC differed by +0.0034 [-0.0081, 0.0148]. The fixed-threshold RA-LL sensitivity difference was **-0.0103 [-0.0355, +0.0113]**, whereas specificity differed by **+0.5309 [0.5211, 0.5403]**. Thus the baseline's apparent reversal sensitivity is coupled to a 64.8% predicted-positive fraction and poor discrimination.

| Arm / condition | AUPRC | AUROC | Sensitivity | Specificity | Predicted + | Brier ↓ | ECE15 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unprotected / clean | 0.5726 | 0.9031 | 0.6362 | 0.9140 | 0.1661 | 0.0994 | 0.0768 |
| Physics-aware / clean | 0.5760 | 0.8927 | 0.6646 | 0.8993 | 0.1828 | 0.1152 | 0.1120 |
| Unprotected / simulated RA-LL | 0.1647 | 0.5383 | 0.6969 | 0.3599 | 0.6483 | 0.4917 | 0.5477 |
| Physics-aware / simulated RA-LL | 0.5683 | 0.8901 | 0.6866 | 0.8908 | 0.1933 | 0.1195 | 0.1253 |
| Huang→clean ResNet* / clean | 0.5721 | 0.9025 | 0.6358 | 0.9141 | 0.1659 | 0.0997 | 0.0772 |
| Huang→clean ResNet* / simulated RA-LL | 0.5811 | 0.9066 | 0.6548 | 0.9117 | 0.1708 | 0.0985 | 0.0769 |

*Huang rows are a post-hoc exploratory comparison on the same independent HEEDB cohort, not an independent test of the method-selection decision. All operating points use unchanged Ningbo validation thresholds; each value is a mean of three seed-level metrics. No HEEDB recalibration was done.*

![Figure 2: HEEDB ranking and operating-point tradeoffs](figures/figure2_heedb_tradeoffs.png)

*Figure 2.* AUPRC, sensitivity, specificity and ECE15 under clean and exact simulated RA-LL. Huang is an exploratory detect→predicted-correct→clean-ResNet operational chain; its comparator-specific false corrections and the physics-aware sensitivity/calibration tradeoff remain visible.

Core SB and STach are supportive but distinct outcomes. Under simulated RA-LL, baseline versus physics-aware AUPRC was 0.8853 versus 0.9757 for SB and 0.5953 versus 0.9485 for STach; RA-LL sensitivity/specificity were 0.9676/0.8096 versus 0.9971/0.8526 for SB, and 0.8650/0.8143 versus 0.9962/0.9005 for STach. Under stricter physician-code definitions, TWA RA-LL AUPRC was 0.1004 versus 0.3156 and SB 0.7773 versus 0.8742; strict-TWA sensitivity favored baseline by 0.0339 [0.0023, 0.0662] in the reverse contrast. These secondary results cannot resolve code-meaning ambiguity.

### Published detector comparator: strong, but exploratory

The Huang detector correctly called 9,701/10,000 clean ECGs and 9,919/10,000 simulated RA-LL ECGs. It therefore made 299 false corrections on clean ECGs and 81 wrong fault calls on reversed ECGs; the true swap was never an oracle for correction. In the operational chain, simulated RA-LL TWA AUPRC was 0.5811 and AUROC 0.9066. Physics-aware minus Huang AUPRC was -0.0128 [-0.0280, +0.0019], **an unresolved difference with no established AUPRC disadvantage or equivalence**. Physics-aware minus Huang sensitivity was +0.0318 [+0.0189, +0.0457], but specificity was -0.0209 [-0.0246, -0.0172], AUROC -0.0166 [-0.0204, -0.0124], and Brier +0.0210 [+0.0188, +0.0232]. These are descriptive post-hoc contrasts under each arm's frozen threshold.

![Figure 3: Paired Huang contrasts](figures/figure3_heedb_paired_contrasts.png)

*Figure 3.* Exploratory physics-aware minus Huang paired patient-bootstrap contrasts on the same 10,000 HEEDB patients (1,000 replicates; 95% percentile intervals). AUPRC crosses zero; the other metrics reveal opposing tradeoffs. Different horizontal scales are intentional and labeled.

## 4. Discussion and limitations

The key methods finding is not that masking is useless, but that consequence-count-matched missingness fails to model a coherent terminal wiring error. Exact simulated reversals expose a large ranking gap for the tested clean-trained architecture; physics-aware augmentation preserves much of diagnostic ranking across the two source test sets and a separate HEEDB physician-code proxy setting. The Huang operational pipeline also repairs the failure and is a serious comparator, with no established AUPRC inferiority. In HEEDB it has better AUROC, specificity and calibration than the physics-aware model under simulated RA-LL, while the physics-aware model has higher frozen-threshold sensitivity. None of these observations proves clinical benefit.

Important boundaries are: (1) cable faults were algebraic simulations, not verified real acquisition errors; they do not cover impedance, motion, open circuits or imperfect placement; (2) HEEDB's overread 12SL code combinations and negative-code choices may not mean the same thing as Chapman/Ningbo SNOMED labels; (3) historical Chapman/Ningbo experiments and the Huang comparison have different timing/freeze status, and Huang was chosen after viewing HEEDB outcomes; (4) one architecture and three seeds do not prove broad model generality; (5) unchanged thresholds are transportability probes, not clinically selected operating points, and clean physics-aware calibration was worse than clean baseline in HEEDB; (6) the intervals are not equivalence or multiple-comparison-adjusted tests; (7) the HEEDB confirmation sample was new and patient-disjoint, but came from the same Emory repository; (8) no prospective workflow, clinician adjudication of actual cable swaps or patient outcome was assessed. A stronger claim requires the evidence listed in [READINESS_DECISION.md](READINESS_DECISION.md), not retrospective wording changes.

## Reproducibility and data availability checklist

- [x] Frozen protocol and code hashes for Chapman/Ningbo/HEEDB and comparator independently verified; see [verification report](VERIFICATION_REPORT.md).
- [x] Patient-disjoint HEEDB confirmation; one ECG per patient, 10,000 evaluation patients; no overlap with original 10,000 or 23 readiness patients.
- [x] Matched three seeds, exact 12 × 1,000 HEEDB input, physics-library terminal transformation, fixed Ningbo normalization/thresholds, 792-mask Chapman/Ningbo control.
- [x] Outcome source, core/strict mappings, prevalence, fixed-threshold operating points, calibration and 1,000 paired bootstraps reported.
- [x] Aggregate-only [figure data](FIGURE_DATA.csv) and [figure hashes](FIGURE_MANIFEST.json); no restricted waveform or row-level prediction in this package.
- [ ] Clinician sign-off on HEEDB code semantics and negative/ambiguous-code policy before external submission; questions below.
- [ ] Institutional/BDSP data-use and code-release review before public repository or venue upload; see [citation/data-use note](DATA_USE_CITATION.md).
- [ ] Final venue-specific formatting, author contributions, ethics/access wording, conflict-of-interest/funding statements, and coauthor approval.

## Exact HEEDB label questions for clinical review

1. Does 12SL physician-overread code **1140 alone** denote a definite T-wave abnormality, and do codes **1145/1150/1151/1160/1161/1170/1171/1172/1180/1181** belong in the same positive construct? Which encode nonspecific, possible, regional, age/sex-adjusted or serial-comparison statements?
2. For an ECG bearing a T-wave-related physician code **outside** that core set, is classifying it as negative clinically misleading? Which codes should instead be excluded or marked indeterminate before a future protocol?
3. Are codes **21 and 24** both definite sinus bradycardia, and is code **21** an appropriate strict subset? Are there rhythm-context exclusions such as atrial arrhythmia or pacing?
4. Does code **23** uniquely identify sinus tachycardia, and can it coexist with non-sinus rhythm codes that should change the interpretation?
5. Can `codes_physician` retain unedited 12SL software suggestions, or does its presence reliably identify a physician-endorsed overread? What is the audit trail for amended reports?
6. Is there any defensible semantic crosswalk from these 12SL constructs to SNOMED TWA 164934002, SB 426177001 and STach 427084000, or must they remain intentionally non-equivalent proxy endpoints?
7. Are missing/empty overread fields and mixed positive/negative statement bundles interpreted as unlabeled rather than true negatives in the present selection rule?

## Citations and restricted-data note

For HEEDB, cite both the [version 5.0 BDSP dataset DOI](https://doi.org/10.60508/rv6h-7d10) and [Koscova et al., *Scientific Data* 2026](https://doi.org/10.1038/s41597-026-06861-9); the BDSP page also explicitly requests its linked [original medRxiv publication](https://doi.org/10.1101/2024.09.27.24314503). The v5.0 [credentialed-data license](https://www.bdsp.io/content/heedb/view-license/5.0/) and [DUA](https://www.bdsp.io/content/heedb/view-dua/5.0/) govern HEEDB files, not the article's open-access license. No HEEDB waveform, patient-level manifest/prediction or record-level example is included or proposed for public release. This draft and its figures have not been submitted or shared externally.
