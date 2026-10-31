# ECG cable reversal research report

Dastan Musrepov | Prepared for Sarvesh Gharat and the research team | 7 October 2026

## 1 Purpose and current conclusion

I investigated whether an ECG classifier that appears robust when leads are missing can still fail when electrode cables are connected to the wrong limbs. I then tested whether including electrically correct cable swaps during training protects the classifier, and compared this approach with a published detector that finds the swap and corrects the signal before diagnosis.

The completed experiments support a clear finding. In our clean-trained ResNet, a simulated right-arm/left-leg cable reversal substantially damaged T-wave-abnormality ranking on Chapman and Ningbo, even though all twelve input channels remained present. The damage was larger than the average damage caused by masking five channels. Training with three physically valid cable reversals recovered most of the lost ranking performance. The result transported to a new 10,000-patient HEEDB/Emory sample without training or tuning on HEEDB.

The published Huang detect-and-correct pipeline also repaired the failure. We did not establish that our method has better diagnostic AUPRC than that pipeline. Our simpler pipeline was faster in a controlled CPU implementation benchmark, while the HEEDB comparison showed different sensitivity, specificity and calibration tradeoffs. All acquisition faults in our experiments were software simulations on real recorded ECGs. We have not tested a cohort with verified real cable mistakes or measured patient benefit, repeat-recording avoidance or financial savings.

This report covers the complete scientific progression of the current ECG project, including the early hypotheses that were rejected, data audits, completed training arms, published-model adaptations, external evaluation, implementation corrections, uncertainty analysis and remaining questions. It includes the completed scientific evidence available through 26 September 2026. Preparing this update did not add new training or scientific experiments.

### Main results at a glance

| Cohort | Baseline clean | Baseline masks | Baseline RA LL | Physics RA LL |
| --- | --- | --- | --- | --- |
| Chapman | 0.6531 | 0.5738 | 0.3278 | 0.6372 |
| Ningbo | 0.6761 | 0.5139 | 0.1822 | 0.6665 |
| HEEDB independent | 0.5726 | Not exhaustive | 0.1647 | 0.5683 |

All values above are AUPRC for T-wave abnormality, averaged over three seed-level metric values. HEEDB uses physician-overread code proxies; those labels are related to, but not identical with, the SNOMED labels in Chapman and Ningbo. AUPRC is a ranking metric, not classification accuracy.

The independent HEEDB reversal gain was +0.4036 AUPRC, or 40.36 AUPRC points, with a paired 95% patient-bootstrap interval of 0.3785 to 0.4255. The HEEDB physics-aware-minus-Huang difference was -0.0128, with an interval of -0.0280 to +0.0019. The second interval does not establish superiority or equivalence. [S1, S8, S10]

## 2 The problem and why it matters

An ECG records electrical activity associated with the heart. A standard recording uses ten electrodes: right arm RA, left arm LA, left leg LL, right leg RL and six chest electrodes. The recording device combines their potentials into twelve displayed leads. These twelve time series become the model's twelve input channels.

Missingness and cable reversal create different inputs. A missing lead supplies no useful waveform in that channel. A cable reversal supplies a waveform that can look plausible, but the lead name no longer has its expected electrical meaning. For example, a model may learn that a certain T-wave shape in lead I is associated with a diagnosis. A reversal can invert that shape or substitute another lead's electrical view, creating a convincing but misleading input.

Random lead masking is useful for studying reduced-lead devices and unavailable channels. It cannot, by itself, demonstrate tolerance to a complete recording with incorrect cable connections. We use masking as a comparison to identify this difference. We do not argue that existing reduced-lead research should have solved every acquisition-error problem.

Cable mistakes occur in practice, but their frequency depends on setting and how errors are confirmed. Rudiger et al. examined 838 ECGs: switched electrodes were confirmed in 3 of 739 outpatient ECGs and 4 of 99 intensive-care ECGs, approximately 0.4% and 4.0%. These are findings from one 2007 study, not an estimate for our datasets or every modern hospital. [R1]

If a recording error is recognized, checking cables and repeating the ECG can be appropriate. The research motivation is that an automatic diagnosis may be produced before the error is recognized. Robust classification could provide a useful safeguard in such a workflow, but preserving the diagnosis does not verify that acquisition was correct. A robust classifier should not be presented as a reason to ignore a known recording fault.

We did not measure the cost or time of repeating an ECG. We also did not measure the incidence of actual miswiring in Chapman, Ningbo or HEEDB. Consequently, there is no defensible estimate of money saved, clinician time saved or avoided clinical errors from this study. The measured practical advantage is lower computational latency in one implementation; its financial value remains unmeasured.

## 3 Concrete research formulation

The main question is: Does an electrically exact simulated limb-cable reversal produce a diagnostic ranking failure that is underestimated by a five-lead masking benchmark, and can training on the reversal family reduce that failure while preserving performance on clean ECGs?

The comparison has three parts. First, I evaluated a clean-trained classifier on clean ECGs, exact RA-LL reversals and all 792 ways of masking five of twelve displayed leads. Second, I compared clean training with masking, reversal augmentation and a combined arm on Chapman, then locked the physics-aware method for a Ningbo follow-up. Third, I tested transportability to HEEDB and compared direct robust diagnosis with a published detect-and-correct pipeline.

Each diagnostic model receives a 12 by 1,000 array and produces three scores: T-wave abnormality TWA, sinus bradycardia SB and sinus tachycardia STach. The primary outcome is TWA. SB and STach characterize whether the effect also occurs for rate-related labels. Labels can coexist, so the task is multilabel prediction rather than choosing exactly one diagnosis.

For the masking comparison, I compute AUPRC separately for every fixed mask, then average the 792 metric values. I subtract the AUPRC under RA-LL reversal from that average. A positive contrast means the reversal is more damaging than the average five-lead mask. This calculation does not average the mask prediction vectors to make an ensemble.

For HEEDB, the primary contrast is physics-aware AUPRC minus baseline AUPRC under the exact simulated RA-LL reversal, averaged across three matched seeds. I also report clean performance, sensitivity, specificity, predicted-positive fraction and calibration. These additional quantities show whether improved ranking translates into a useful operating point at the fixed threshold.

Five leads were chosen for the masking panel because RA-LL changes five displayed leads. This matches the number of affected inputs. It does not match lost information, fault probability or clinical severity. That distinction is part of the interpretation, not a reason to call the two conditions electrically equivalent.

## 4 Data sources and analytical populations

### Chapman and Ningbo

The waveform source was the official PhysioNet Computing in Cardiology Challenge 2021 release, version 1.0.3. Chapman-Shaoxing and Ningbo are two constituent sources in that release. I did not use the challenge's hidden competition test set. I created project-specific training, validation and test splits within the public data, using the saved split manifests. [R2, R3]

| Source | Source records | Analytical records | Train | Validation | Test |
| --- | --- | --- | --- | --- | --- |
| Chapman | 10,247 | 10,234 | 7,163 | 1,021 | 2,050 |
| Ningbo | 34,905 | 34,899 | 24,430 | 3,488 | 6,981 |
| HEEDB first cohort | Selected subset | 10,000 patients | None | None | 10,000 |
| HEEDB confirmation | New disjoint subset | 10,000 patients | None | None | 10,000 |

Chapman included 10,247 source recordings. Thirteen duplicate waveform exports were collapsed under a documented amendment, leaving 10,234 canonical recordings. Their metadata and endpoint labels agreed. Ningbo included 34,905 source recordings; three waveform-duplicate pairs had conflicting age, sex or full diagnosis metadata. All six records were excluded before splitting under a separately frozen amendment, leaving 34,899 recordings. No exact waveform or record-ID overlap remained across the analytical splits. [S2, S3]

| Test source | TWA positive | SB positive | STach positive | Total |
| --- | --- | --- | --- | --- |
| Chapman | 377 (18.39%) | 780 (38.05%) | 314 (15.32%) | 2,050 |
| Ningbo | 1,034 (14.81%) | 2,534 (36.30%) | 1,138 (16.30%) | 6,981 |

The source labels used SNOMED codes 164934002 for TWA, 426177001 for SB and 427084000 for STach. The models received waveforms only; age and sex were used for audits or descriptive analyses rather than diagnostic input features. The three target labels were not a comprehensive evaluation of all cardiac diagnoses.

The public Challenge files do not provide an independently verified patient identifier for every analytical recording. Record-ID separation and waveform deduplication reduce obvious leakage, but cannot exclude different recordings from the same person across splits. Historical bootstrap analyses treated canonical recordings as the observation units. Referring to those intervals as unequivocally patient-level would overstate the available identifiers. HEEDB provides actual patient-linked selection, which addresses this limitation for the later external evaluation.

### HEEDB and access

After receiving credentialed BDSP access, I used HEEDB version 5.0, institution I0006, the Emory source. The full database includes many more ECGs than we downloaded. Our study used a selected subset, not training on the entire database. Access was obtained through the required account, research application, reference, training and data-use steps; data transfer used the approved AWS access point. [R4]

The external selection joined waveform metadata with acquisition-time diagnosis data and used codes_physician, the physician-overread field. It did not merge codes_software, reprocessed v24 codes or ICD labels into the primary outcomes. Rows with missing patient IDs and empty physician-code fields were excluded by the selection rule. One recording per patient was selected with a label-blind hashing procedure.

The first 10,000-patient sample supplied the original v1 evaluation and its corrected v2 rerun. A later new 10,000-patient sample excluded those patients and the 23 readiness patients. All three patient sets had zero pairwise overlap. The two 10,000-patient cohorts remain separate analyses; they were not pooled to produce the primary result. [S7, S8]

The new confirmation downloaded 20,000 waveform/header objects, approximately 1.096 GiB in total. No HEEDB model was fitted, recalibrated or threshold-tuned. The saved Ningbo baseline and physics-aware models were transported unchanged. Neither ECGFounder nor the Huang detector was trained on our HEEDB subset in this project.

### HEEDB outcome definitions

| Outcome | Physician codes | First 10000 | New 10000 |
| --- | --- | --- | --- |
| Core TWA | 1140, 1145, 1150, 1151, 1160, 1161, 1170, 1171, 1172, 1180, 1181 | 1,524 | 1,456 |
| Core SB | 21 and 24 | 1,665 | 1,637 |
| Core STach | 23 | 673 | 615 |
| Strict TWA | 1140 | 770 | 727 |
| Strict SB | 21 | 1,448 | 1,453 |

The broad core-TWA code set and the strict 1140-only definition produce different class prevalence and sensitivity. They are prespecified proxy constructs. We cannot silently equate them with the Chapman/Ningbo SNOMED label. An independent clinician still needs to review the exact meaning of ambiguous, serial-comparison and mixed T-wave statements and whether some current negatives should be treated as indeterminate. [S7, S8, S12]

## 5 What the files and signals look like

The public Chapman/Ningbo recordings contain a .mat waveform and a .hea header. The waveform is a numerical matrix, not a photograph of an ECG page. The header provides sampling frequency, signal gains, lead names and coded diagnoses. A 10-second recording at 500 Hz has 5,000 values per lead.

HEEDB I0006 uses .dat waveforms with .hea headers. Original recordings are 10 seconds at 250 or 500 Hz. Reading the file correctly requires respecting the header's units, gain, channel order and sampling rate. A model can accept an array of the wrong length without crashing; that does not mean its input matches training.

The canonical lead order is I, II, III, aVR, aVL, aVF and V1 through V6. The displayed limb leads are dependent: III equals II minus I, aVR equals minus the sum of I and II divided by two, aVL equals I minus II divided by two, and aVF equals II minus I divided by two. The physics library represents eight independent measurements: I, II and the six chest leads.

RA, LA and LL are represented in a convenient algebraic coordinate system with RA set to zero, LA represented by I and LL represented by II. This does not mean the physical RA voltage is actually zero. It is a reference choice that preserves potential differences. Wilson's central terminal is the mean of the three limb-electrode potentials; permuting those three terminals leaves this mean unchanged, explaining why V1-V6 remain invariant for our three tested limb swaps. [R5, S13]

## 6 Signal manipulations and electrical validation

### Clean condition and five lead masking

The clean condition retains the recorded signal after the fixed processing steps. For masking, selected displayed leads are replaced with their training-set mean. After training-set normalization, the masked values become zero. This is a numerical missing-input sentinel, not a measured response of a detached electrode or a claim that a broken cable physically outputs an average waveform.

I enumerated every combination of five displayed leads, which gives 792 fixed masks. For each model and test cohort, I evaluated clean input, exact RA-LL input and the 792 masks: 794 conditions per model. A given mask is fixed across the recordings in that condition; the reported mask mean averages condition-level AUPRC values.

### Exact cable reversals

| Displayed lead | RA LA swap | RA LL swap | LA LL swap |
| --- | --- | --- | --- |
| I | -I | -III | II |
| II | III | -II | I |
| III | II | -I | -III |
| aVR | aVL | aVF | aVR |
| aVL | aVR | aVL | aVF |
| aVF | aVF | aVR | aVL |
| V1 through V6 | Unchanged | Unchanged | Unchanged |

The physics library first permutes limb terminal connections, then derives the displayed leads from the transformed independent signals. RA-LL changes I, II, III, aVR and aVF. It leaves aVL and all six chest leads unchanged. RA-LA and LA-LL follow different patterns; treating them as arbitrary channel shuffles would be wrong.

These transformations simulate exchanging cables connected to the same limb-electrode locations, under conventional simultaneous twelve-lead derivation. They do not simulate moving a chest electrode to another anatomical position, impedance changes, motion, a detached electrode, device-specific filtering faults or RL swaps. Those processes require additional physical measurements or a different validated model.

Unit tests checked Einthoven and Goldberger identities, the terminal permutations, unchanged chest leads and reversal consistency. Later HEEDB adapter tests compared all twelve transformed channels with the common physics library, including aVR and aVF, and checked the expected input length. Detector correction was selected from the detector's predicted class, never the known simulation label. [S13]

## 7 Processing pipeline

For the small ResNet, the fixed public-data loader takes every fifth sample from each 500 Hz waveform, producing 1,000 values per lead over 10 seconds, an effective 100 Hz input. This is direct decimation; no separate antialiasing filter is implemented in that historical loader. Possible aliasing is therefore a preprocessing limitation. The completed arms use the same loader, so a new filtering choice would require a separate experiment rather than rewriting their results.

Training-set statistics provide one mean and standard deviation per displayed lead. Training, validation and test data use those saved statistics. Reversal augmentation occurs before normalization so that the transformation respects electrical relationships in raw units. Masked channels use the corresponding training mean.

For HEEDB, WFDB physical millivolts are converted to microvolt-scale values consistent with the public model input. Original 250 Hz signals are resampled to 500 Hz, yielding 5,000 samples. The waveform is then sampled at every fifth position to exactly 12 by 1,000, transformed with the same electrical library and normalized with the relevant saved Ningbo statistics. The corrected evaluator checks input shape and finite model outputs.

ECGFounder and Huang use their own published preprocessing. I did not force them through the small ResNet loader. Their adapters preserve the intended input representation before their pretrained weights are applied. The resulting diagnostic comparisons must therefore be interpreted as comparisons of complete adapted pipelines, rather than architecture-only experiments. [S6, S9, S10]

## 8 Model architecture and training

The primary classifier is a small one-dimensional ResNet. A convolutional stem receives twelve channels and has 32 feature channels. Four residual blocks use widths 32, 64, 128 and 256, with seven-sample convolution kernels, group normalization and GELU activations. Later blocks downsample the time axis. Global average pooling, dropout of 0.2 and a linear layer produce three diagnostic logits, followed by sigmoid scores at inference. It has no explicit cable-error detector.

Training uses AdamW, learning rate 0.001, weight decay 0.0001, batch size 64 and at most 30 epochs. The loss is multilabel binary cross-entropy with positive-class weights derived from training counts and capped at 20. The best checkpoint is selected using validation TWA AUPRC, with early stopping after five epochs without improvement. CUDA mixed precision was used on the RTX 5070 Laptop GPU. [S4, S5]

Each completed principal comparison uses three seeds: 104729, 130363 and 155921. A seed changes initialization and training randomness. Three seeds measure some training variability; they do not represent three populations or prove invariance across all architectures. Reported means average the three metric values, rather than combining three prediction vectors into an ensemble.

| Setup | Clean ECG | Five lead mask | Exact reversal | Where completed |
| --- | --- | --- | --- | --- |
| Baseline | 100% | 0% | 0% | Chapman and Ningbo |
| Random masking | 50% | 50% | 0% | Chapman |
| Physics aware | 50% | 0% | 50% | Chapman and Ningbo |
| Combined | 50% | 25% | 25% | Chapman |

All four setups were compared on Chapman. The physics-aware arm was the prespecified primary mitigation, while masking and combined were secondary. The Ningbo mitigation follow-up trained only the locked physics-aware arm; it did not repeat a full four-arm method search. Ningbo baseline and augmented models were trained on Ningbo training records, so this was replication of the method on another source. Actual no-refitting model transfer occurred later when those Ningbo checkpoints were evaluated on HEEDB.

Before test evaluation, each trained unit had to pass a validation TWA AUROC of at least 0.75, AUPRC of at least 0.45, finite-loss/prediction checks and artifact-integrity checks. Frozen protocols defined success criteria before that follow-up's evaluation. Historical test outcomes were already known for some follow-ups, so the complete project is not a single untouched preregistered study.

| Chapman training | Clean TWA | RA LL TWA | Mean masks TWA |
| --- | --- | --- | --- |
| Clean baseline | 0.6531 | 0.3278 | 0.5738 |
| Random masking | 0.6525 | 0.2797 | 0.6340 |
| Physics aware | 0.6641 | 0.6372 | 0.5679 |
| Combined | 0.6586 | 0.6314 | 0.6312 |

This full Chapman arm table is computed from the saved RESULTS_BY_ARM_SEED.csv. It includes the secondary masking and combined arms rather than omitting them from the project history. The detailed metrics include SB, STach, operating points and calibration. The primary follow-up decision selected physics-aware training by its frozen role and gate, not by selecting the best test arm after observing this table. [S4]

Random-masking training improved average-mask TWA AUPRC to 0.6340, but its RA-LL AUPRC was only 0.2797. Combined training reached 0.6314 under RA-LL and 0.6312 under masking, compared with 0.6372 and 0.5679 for the physics-aware arm. These Chapman point estimates suggest a useful tradeoff, but do not establish a superior combined method. The combined arm has no completed locked Ningbo mitigation evaluation in this project.

## 9 Metrics thresholds and uncertainty

AUPRC summarizes how a model ranks positive cases while balancing precision and recall. It is especially informative when the positive class is relatively uncommon. Its value depends on prevalence, so AUPRC from two cohorts should not be treated as an architecture ranking without considering labels and population. The analysis computes sklearn average precision, referred to as AUPRC in the saved reports.

AUROC measures ranking of positive versus negative cases across thresholds. Sensitivity is the fraction of actual positives classified positive at a chosen threshold. Specificity is the fraction of actual negatives classified negative. Predicted-positive fraction is the proportion of all records flagged positive. A model that flags nearly everyone can have high sensitivity but poor precision and specificity.

Each seed's threshold for each endpoint was the 90th percentile of scores among clean validation negatives, targeting approximately 90% validation specificity. Those thresholds remained fixed during fault tests and HEEDB transfer. The test population need not retain 90% specificity. Huang's downstream classifier uses the corresponding clean baseline threshold; physics-aware models retain their own validation thresholds.

Calibration asks whether a predicted score of, for example, 0.8 corresponds to a roughly 80% positive frequency. Brier score measures squared probability error. ECE15 groups scores into fifteen equal-width bins and compares mean confidence with positive frequency. Lower values are better, but a single calibration summary does not prove reliable clinical probabilities.

Each principal uncertainty analysis used 1,000 paired bootstrap resamples. The same resampled observations were used for both compared arms, conditions and seeds, preserving pairing. Chapman/Ningbo used canonical-record units. HEEDB used one ECG per identified patient and patient-level resampling. The analysis calculated condition-level metrics inside each replicate, then seed-mean contrasts and percentile 95% intervals.

An interval excluding zero supports a directional contrast under this analysis. An interval containing zero does not show that two approaches are equivalent. A formal equivalence or noninferiority claim would require a justified margin and a prespecified analysis. Reported intervals are not adjusted for all secondary comparisons, and population-resampling intervals do not capture every form of training or site uncertainty.

## 10 Early tests and changes in direction

The first feasibility phase required a non-PTB confirmation source with authoritative patient IDs. That requirement blocked the initial design because public Challenge sources did not establish the requested patient-linked inference. Later protocols narrowed the claim to canonical-record inference with explicit patient-identity limitations. This was a design amendment, not new proof that every record belonged to a different patient.

The initial Chapman structured-missingness pilot focused on the V1-V4 precordial block. The ingestion audit first stopped on thirteen exact duplicate exports. A documented pre-split amendment allowed canonicalization of duplicates whose metadata agreed, after which training and evaluation proceeded.

The subsequent interpretation audit found two problems in the early missingness result. The automatic decision had used the absolute value of a signed contrast, losing whether the hypothesized block was actually worse. In addition, seventy nominal independent-measurement masks collapsed to fifty unique prediction outcomes. V1-V4 removal was less damaging than its average comparator, so that experiment did not establish the proposed mechanism. Its result was retained as mixed/negative evidence rather than presented as success. [S14]

I then examined electrically exact cable reversals on the existing Chapman model. In the exploratory single-seed analysis, clean TWA AUPRC was 0.6472; RA-LA gave 0.4894, RA-LL 0.2753 and LA-LL 0.6292. The average five-lead-mask AUPRC was 0.5835. RA-LL was therefore a stronger, physically specified candidate failure than the original V1-V4 missingness hypothesis. That discovery preceded the later confirmation protocol; the selected reversal was not discovered on a wholly untouched Chapman test set. [S15]

A three-seed Chapman stress analysis then reproduced the RA-LL gap. Ningbo was reserved until its protocol was frozen. Ningbo v1 stopped during its duplicate audit; v2 excluded the six metadata-conflicting duplicate records before splitting and confirmed the reversal gap. The later physics-aware M1 and M2 experiments addressed mitigation after those baseline outcomes had been observed.

An additional TimePatchTransformer architecture experiment was implemented and trained/evaluated on Chapman. Its retained status ends at 794 of 794 conditions for the last unit, with zero completed bootstrap replicates and no final M3 scientific decision or complete summary. I therefore do not count it as completed architecture replication or quote it as evidence of generality. It is an unfinished branch in the current evidence inventory, not an active experiment that this report resumed.

## 11 Completed public data results

### Baseline failure under reversal

| Source | Label | Clean AP | Mask AP | RA LL AP | Mask minus swap 95% CI |
| --- | --- | --- | --- | --- | --- |
| Chapman | TWA | 0.6531 | 0.5738 | 0.3278 | 0.246 [0.210, 0.280] |
| Chapman | SB | 0.9993 | 0.9988 | 0.9938 | 0.005 [0.003, 0.007] |
| Chapman | STach | 0.9856 | 0.9833 | 0.7557 | 0.227 [0.187, 0.273] |
| Ningbo | TWA | 0.6761 | 0.5139 | 0.1822 | 0.331 [0.311, 0.350] |
| Ningbo | SB | 0.9960 | 0.9945 | 0.9568 | 0.038 [0.032, 0.045] |
| Ningbo | STach | 0.9856 | 0.9740 | 0.7020 | 0.271 [0.250, 0.294] |

The baseline mean-mask-minus-RA-LL TWA contrast was approximately +0.246 on Chapman and +0.331 on Ningbo, with 95% intervals 0.210 to 0.280 and 0.311 to 0.350. Every TWA seed showed the same direction. These are 24.6 and 33.1 AUPRC points relative to the average masking condition, rather than percentage changes in accuracy. [S1, S2, S3]

The STach effect was also large, while SB was smaller. An ideal limb-cable swap preserves underlying beat timing. The STach ranking loss is therefore consistent with the classifier depending on waveform morphology or lead configuration in addition to rate. Cached score shifts and error transitions show that outputs changed, but they do not identify a particular shortcut or establish the causal feature. A feature-attribution or targeted representation experiment remains necessary for that stronger explanation.

[Aggregate Figure 1 in the Word and PDF report]

Figure 1. TWA ranking on Chapman and Ningbo under clean input, the mean of all 792 five-lead masks, and exact simulated RA-LL. The plotted values are aggregate saved results; this figure contains no patient waveform.

### Recovery after physics aware training

On Chapman, physics-aware training increased RA-LL TWA AUPRC from 0.3278 to 0.6372, a gain of 0.3094. Clean AUPRC changed from 0.6531 to 0.6641 and average-mask AUPRC from 0.5738 to 0.5679. On Ningbo, RA-LL AUPRC increased from 0.1822 to 0.6665, a gain of 0.4843. Clean AUPRC changed from 0.6761 to 0.6869 and mean-mask AUPRC from 0.5139 to 0.5375. [S4, S5]

The paired reduction of the mask-minus-reversal contrast was 0.3145 on Chapman, interval 0.2771 to 0.3508, and 0.4598 on Ningbo, interval 0.4348 to 0.4839. This is a difference of contrasts, not the same quantity as the direct reversal AUPRC gain. Both frozen mitigation gates passed.

The method has a simple rationale: the training distribution now includes signals produced by valid cable permutations while keeping the diagnostic label. The network is encouraged to use evidence that remains predictive across those views. This rationale does not prove which features changed or that the method handles every acquisition error. It explicitly teaches the same reversal family later tested.

## 12 ECGFounder adaptation and findings

ECGFounder is a published general-purpose ECG model built on more than ten million recordings, with an open pretrained checkpoint. Its official 150-class output vocabulary has no exact match for our TWA SNOMED endpoint. NONSPECIFIC T WAVE ABNORMALITY is a narrower/different statement and was not substituted as an exact label. Direct sinus-bradycardia and sinus-tachycardia outputs were treated as separate name-matched exploratory results. [R6, S6]

For an exact-label diagnostic comparison, I froze ECGFounder's pretrained backbone, extracted its 1,024-dimensional features and trained three binary logistic-regression heads for our endpoint definitions using each source's existing training split. L2 regularization was chosen from 0.01, 0.1 and 1 by clean validation AUPRC. The pretrained encoder was not fine-tuned. This is a frozen-feature adaptation, not a complete reproduction of the authors' best finetuned system.

Input handling followed the pinned authors' preprocessing: physical millivolts at 500 Hz, NaN samples replaced by zero according to the source loader, exact reversal before filtering, a 50 Hz notch, a 0.67-40 Hz fourth-order bandpass, a 201-sample median baseline step and global z-score. Header-path differences and NaN records were audited rather than silently dropped. All 45,133 analytical public records resolved in v2.

| Source | Label | Founder clean | Founder RA LL | ResNet clean | ResNet RA LL |
| --- | --- | --- | --- | --- | --- |
| Chapman | TWA | 0.626 | 0.316 | 0.653 | 0.328 |
| Chapman | SB | 1.000 | 1.000 | 0.999 | 0.994 |
| Chapman | STach | 0.995 | 0.902 | 0.986 | 0.756 |
| Ningbo | TWA | 0.629 | 0.209 | 0.676 | 0.182 |
| Ningbo | SB | 0.998 | 0.965 | 0.996 | 0.957 |
| Ningbo | STach | 0.991 | 0.971 | 0.986 | 0.702 |

TWA ranking deteriorated under reversal in both sources. The 95% interval for the TWA clean-minus-reversal AUPRC drop was 0.256 to 0.365 on Chapman and 0.391 to 0.448 on Ningbo. On Ningbo, ECGFounder's TWA drop was smaller than the ResNet's by 0.074, interval -0.100 to -0.048 when defined as Founder drop minus ResNet drop. The model still had a substantial TWA failure. Its STach rankings were much more stable than the small ResNet's, showing that robustness is endpoint- and pipeline-dependent.

The pretrained weights can be evaluated without downloading the full pretraining database; published preprocessing and label adaptation are still necessary. This run used previously examined public test cohorts, and possible public pretraining overlap was not independently excluded. It therefore supports an exploratory stress finding, not a claim that our small model outperformed all ECG foundation models. I did not train physics-aware ECGFounder or evaluate its diagnostic pipeline on the HEEDB confirmation cohort.

## 13 Published detector and correction comparison

Huang et al. published a lightweight lead-misplacement detector and released code and weights. Its endpoint is identifying cable/lead errors, rather than predicting TWA. I used the unchanged limb detector as part of an operational diagnostic pipeline: ECG, detector, predicted correction, saved clean ResNet, diagnosis. This makes it relevant to our exact downstream problem without pretending that diagnostic AUPRC is the metric originally reported in the Huang paper. [R7, S9]

The adapter filters the 500 Hz ECG with a fifth-order 0.5-35 Hz Butterworth filter, resamples to 120 Hz and applies within-record, within-lead z-scoring. The detector receives leads III, aVR and V6 as 1,200 by 3 input and predicts clean, RA-LA, RA-LL or LA-LL. The predicted argmax class chooses an inverse transformation. The known simulation fault is not used to choose deployed correction.

I also retained an oracle correction arm as an implementation upper bound, where the true fault is known. That arm cannot be used as a practical method or combined with the predicted-correction scores to claim deployable performance.

| Ningbo TWA | Baseline AP | Physics AP | Huang chain AP | Physics minus Huang CI |
| --- | --- | --- | --- | --- |
| RA LA | 0.2613 | 0.6694 | 0.6652 | [-0.0113, +0.0204] |
| RA LL | 0.1822 | 0.6665 | 0.6636 | [-0.0133, +0.0187] |
| LA LL | 0.6338 | 0.6691 | 0.6735 | [-0.0194, +0.0098] |
| Worst fault | 0.1822 | 0.6665 | 0.6636 | [-0.0129, +0.0169] |

On Ningbo, physics-aware-minus-Huang TWA AUPRC differences were +0.0041 for RA-LA, +0.0029 for RA-LL and -0.0044 for LA-LL. Their 95% intervals all contained zero. The worst-fault contrast was +0.0029 with interval -0.0129 to +0.0169. This provides no established diagnostic superiority or equivalence. The detector correctly left 98.51% of clean Ningbo ECGs unchanged and had RA-LL recall of 98.11%.

The error audit found 239 missed-reversal instances, 217 wrong-reversal-type instances and 104 false corrections of clean ECGs across the three fault conditions and clean condition. These are condition-record instances, not 560 distinct patients. On the wrong-type subset, pooled TWA decision-error rates across seeds were 5.07% for the physics-aware model and 29.80% for the correction chain. This is a descriptive, selected error-subset analysis rather than a population-level superiority result. It helps explain how a detector mistake can propagate into a diagnosis.

Both approaches protect diagnostic ranking, but they expose different outputs. The detector provides an explicit error type and supports correction or re-recording workflows. The physics-aware diagnostic model avoids a separate detector but does not tell the operator which cable was wrong. The choice depends on the operational purpose as well as the diagnostic metric.

## 14 Runtime and deployment measurement

The initial deployment audit timed components and used a conservative two-process memory sum. I then performed a stricter integrated request-to-diagnostic-response benchmark rather than adding component durations. The final numbers below refer to that integrated v4 measurement. [S11]

| Integrated CPU measurement | Physics aware | Huang chain |
| --- | --- | --- |
| Warm request median | 10.59 ms | 31.51 ms |
| Warm request p95 | 12.42 ms | 34.16 ms |
| Loading to ready | 5.82 s | 10.98 s |
| Maximum sampled working set | 612.2 MiB | 953.6 MiB |
| Model artifact size | 3.70 MiB | 3.81 MiB |

The warm median ratio was 2.97, and the physics-aware process tree used approximately 36% less sampled working-set memory. The CPU was an AMD Ryzen AI 7 350 on Windows 11. Both frameworks used one thread pinned to logical CPU 0. Seven runs each processed a balanced 128-ECG fixture after warmup. This is a hardware/software-specific comparison of these Python integrations, not the Huang authors' published runtime or a universal speed claim.

Before timing, the fixture outputs were checked against the frozen seed-104729 caches. Threshold decisions agreed 100% for both pipelines, and detector classes agreed 100%; maximum diagnostic probability differences were 0.0004 and 0.0003. The balanced fixture was chosen for benchmarking, not to represent clinical fault prevalence.

The reduction is about 20.9 milliseconds per warm CPU request in this implementation. It is not a measured reduction in patient waiting time or staff work. A production implementation could optimize either pipeline and change the ratio.

## 15 HEEDB v1 failure and implementation correction

The first HEEDB external run appeared to show poor transportability: clean core-TWA AUPRC was approximately 0.158 for baseline and 0.144 for physics-aware, close to the 0.152 prevalence. A detailed audit found that this run had not applied the intended matched pipeline. Two independent implementation errors affected the evaluation. [S7]

First, the ResNet received 5,000 samples instead of the 1,000 samples used in Ningbo training. Global temporal pooling allowed inference to run, but the temporal filters saw the wrong sample spacing and duration in convolution steps. A finite-output smoke test did not catch that scientific mismatch.

Second, the local v1 reversal routine used incorrect aVR and aVF equations. Exact RA-LL requires aVR to become the original aVF and aVF to become the original aVR. The wrong routine was therefore not an electrically exact realization of the stated condition.

The audit recomputed labels, metrics, checkpoint bindings, amplitude conversions and sample-rate paths. Metric arithmetic and label-channel order matched the saved files; a millivolt/microvolt conversion error was ruled out. The problems were the temporal input and reversal implementation. The original v1 artifacts were preserved as an invalid implementation of the intended experiment, rather than being silently replaced.

A separately frozen corrected v2 reused the same first 10,000 patients, labels, six checkpoints, thresholds and 24-mask subset. It corrected only the sample length and reversal implementation and required new tests on all twelve leads. Because the corrections followed observed outcomes on the same cohort, v2 is a post-hoc implementation correction. [S7]

After correction, clean TWA AUPRC was 0.5770 baseline and 0.5715 physics-aware. Under exact RA-LL it was 0.1774 and 0.5664, a paired physics-aware gain of 0.3890, interval 0.3666 to 0.4126. The 24-mask subset mean was 0.5094 versus 0.5135. This subset cannot be reported as an exhaustive 792-mask evaluation on HEEDB.

The technical lesson is that shape, units, sampling and every lead transformation need explicit adapter tests. A process completing successfully and producing finite probabilities is insufficient evidence that it evaluated the intended scientific question.

## 16 Independent HEEDB confirmation

I then evaluated a newly selected 10,000-patient cohort disjoint from the prior 10,000 and the readiness patients. The models, thresholds, outcome definitions and corrected processing were fixed before its model outcomes were examined. The sample contained 1,456 core-TWA positives, a prevalence of 0.1456. [S8]

| Arm and condition | AUPRC | AUROC | Sensitivity | Specificity | Flagged | ECE15 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline clean | .5726 | .9031 | .6362 | .9140 | 16.61% | .0768 |
| Physics clean | .5760 | .8927 | .6646 | .8993 | 18.28% | .1120 |
| Baseline RA LL | .1647 | .5383 | .6969 | .3599 | 64.83% | .5477 |
| Physics RA LL | .5683 | .8901 | .6866 | .8908 | 19.33% | .1253 |

Physics-aware minus baseline RA-LL AUPRC was +0.4036, with a 95% interval of 0.3785 to 0.4255. The seed-level gains were +0.4527, +0.3718 and +0.3863. Clean AUPRC differed by +0.0034, interval -0.0081 to +0.0148. Thus ranking recovery did not require fitting the models to this cohort, and the clean AUPRC difference was not statistically resolved.

At the unchanged thresholds, reversal sensitivity was 0.6866 physics-aware versus 0.6969 baseline, a difference of -0.0103 with interval -0.0355 to +0.0113. Specificity was 0.8908 versus 0.3599, a difference of +0.5309 with interval 0.5211 to 0.5403. The baseline flagged 64.83% of ECGs, while physics-aware flagged 19.33%. This explains why the large ranking gain must not be described as better sensitivity.

[Aggregate Figure 2 in the Word and PDF report]

Figure 2. HEEDB ranking, threshold-dependent decisions and calibration. The Huang rows in this figure are a later exploratory comparison on the same cohort. Core-TWA labels are physician-code proxies, and all cable faults are simulations.

Calibration deserves separate attention. Physics-aware clean ECE15 was 0.1120 versus 0.0768 for baseline, and clean Brier score was also worse. The model's AUPRC was preserved while its probability calibration and some operating-point measures differed. These outputs were not recalibrated on HEEDB.

### Secondary outcomes and stricter definitions

| Proxy outcome | Positives | Baseline clean | Physics clean | Baseline RA LL | Physics RA LL |
| --- | --- | --- | --- | --- | --- |
| Core SB | 1,637 | .9765 | .9751 | .8853 | .9757 |
| Core STach | 615 | .9520 | .9502 | .5953 | .9485 |
| Strict TWA | 727 | .3179 | .3209 | .1004 | .3156 |
| Strict SB | 1,453 | .8838 | .8700 | .7773 | .8742 |

The paired RA-LL AUPRC gains were +0.0904 for core SB, interval 0.0781 to 0.1021, and +0.3532 for core STach, interval 0.3208 to 0.3886. For strict TWA, AUPRC improved by +0.2152, interval 0.1912 to 0.2401, while sensitivity decreased by 0.0339, interval -0.0662 to -0.0023. This unfavorable strict-TWA sensitivity result is part of the evidence and cannot be omitted by focusing only on the broad core definition.

The first and independent samples showed similar ranking recovery, but they came from the same Emory repository and used the same code constructs. Independent patients provide a stronger replication than reusing the corrected cohort. They do not provide another institution, a clinician-adjudicated real-fault cohort or proof of label equivalence.

## 17 HEEDB comparison with Huang

After viewing the independent HEEDB results, I applied the unchanged published Huang detector to that same cohort. This comparison was planned after outcomes were known and is explicitly exploratory. It used predicted correction, not true-fault correction, and ran the corrected waveforms through the saved clean Ningbo ResNets. No new records were selected or downloaded. [S10]

| RA LL TWA pipeline | AUPRC | AUROC | Sensitivity | Specificity | Brier | ECE15 |
| --- | --- | --- | --- | --- | --- | --- |
| Unprotected | .1647 | .5383 | .6969 | .3599 | .4917 | .5477 |
| Physics aware | .5683 | .8901 | .6866 | .8908 | .1195 | .1253 |
| Huang correction | .5811 | .9066 | .6548 | .9117 | .0985 | .0769 |

Huang's downstream RA-LL TWA AUPRC was 0.5811 versus 0.5683 physics-aware and 0.1647 unprotected. The paired physics-aware-minus-Huang AUPRC difference was -0.0128, interval -0.0280 to +0.0019. Its sign varied across seeds. We therefore cannot claim that the physics-aware approach outperformed this published operational baseline in diagnostic ranking.

Huang had better reversal AUROC, specificity and calibration. Physics-aware had higher sensitivity at its own frozen threshold: the paired sensitivity difference was +0.0318, interval 0.0189 to 0.0457. Its specificity difference was -0.0209, interval -0.0246 to -0.0172. These are opposing operating-point tradeoffs and the two arms use their own previously fixed thresholds. The comparison does not establish a superior clinical decision policy.

[Aggregate Figure 3 in the Word and PDF report]

Figure 3. Exploratory paired physics-aware-minus-Huang contrasts on the independent HEEDB cohort. The AUPRC interval includes zero; the remaining metrics show different tradeoffs. The intervals do not establish equivalence.

The detector correctly left 9,701 of 10,000 clean ECGs unchanged and correctly classified 9,919 of 10,000 simulated RA-LL ECGs. It made 299 false corrections on clean ECGs and 81 wrong calls on reversed ECGs. On false-corrected clean ECGs, 14.5% of seed-record diagnostic decisions differed from the clean reference; on the wrong-call reversed subset, 28.8% differed. These percentages are descriptive analyses of selected detector-error subsets, not overall patient error rates.

## 18 Related work and comparison boundaries

The closest operational comparator for the tested fault family is Huang's released detector chained to our diagnostic classifier. ECGFounder is a strong published representation/diagnostic model used to test whether the vulnerability appears in another pipeline. There is no established universal SOTA leaderboard for our exact combination of three diagnoses, the selected public splits, HEEDB proxy labels, exact cable swaps and matched five-lead masks.

Oh et al. introduced lead-agnostic self-supervised ECG learning with random lead masking at CHIL 2022. That work motivates arbitrary lead availability; our masking training is a simple supervised augmentation arm and does not reproduce their self-supervised algorithm. Bo et al.'s adversarial masking pretraining makes the hidden signal regions challenging, rather than constructing terminal reversals. Raghu et al. studied ECG augmentation selection and task dependence. GeoECG perturbs ECGs along Wasserstein geodesics. These are relevant augmentation precedents, but none was implemented as a full comparative model in this project. [R8-R11]

A published LV systolic-dysfunction study, Transparent and Robust Artificial Intelligence-Driven Electrocardiogram Model for Left Ventricular Systolic Dysfunction, also examined simulated lead reversal. Its endpoint differs from TWA/SB/STach. Its existence means our novelty cannot be described as the first study to test an ECG diagnostic model under lead reversal. The narrower contribution is the matched missingness comparison, augmentation evaluation, dataset replication and transparent operational comparison. [R12]

Rjoob et al. studied chest-electrode placement detection. Chest displacement changes anatomical measurement locations and requires a different simulator or measured data. Their endpoint and fault mechanism differ from our three limb-cable swaps. ECG-FM offers another open pretrained model using public ECG sources, including Chapman and Ningbo, so overlap would need to be acknowledged in a public-cohort stress test. Such overlap does not make every comparison impossible, but it limits claims of independent external generalization. Neither model was evaluated in the completed project. [R13, R14]

Other papers discussed during literature review, including 3KG, CLOCS, multimodal ECG representation learning, precordial augmentation, quality-control studies and newer correction abstracts, informed possible directions. They are not completed experimental baselines. Their published scores cannot be compared directly with our AUPRC values when diagnoses, labels, populations, metrics and splits differ.

## 19 Contribution and intuitive explanation

The demonstrated contribution has three parts. The first is a physically specified failure mode that a channel-count-matched masking test underestimates. The second is evidence that a simple training intervention can preserve diagnostic ranking under that fault family across the tested sources, including patient-linked external data. The third is an operational comparison with released detector weights that reveals diagnostic, calibration and computational tradeoffs.

The intuitive explanation is that missingness removes evidence, while a cable reversal changes the meaning of evidence. Training on reversals gives the network examples of the latter. The diagnosis is attached to the underlying recording, so the model learns to retain diagnostic information across several electrically valid views. This is a plausible learning explanation supported by measured recovery, but it does not identify the network's exact internal feature changes.

The algorithmic novelty is modest: the circuit algebra and augmentation principle are established. The study's stronger contribution is the precise comparative design and evidence about when commonly used robustness tests are insufficient for this acquisition fault. A literature review can refine the novelty statement, but cannot manufacture a new training algorithm from these results.

Current evidence supports a carefully written methods or robustness paper. It does not support saying that our model is the overall best ECG classifier, clinically safer than Huang, immune to every placement error, or demonstrably saves money. Higher AUPRC alone cannot establish those claims.

## 20 Engineering incidents and audit trail

Several long runs required technical recovery. Download parts and completed checkpoints were preserved. Process locks and completion artifacts were used to avoid duplicate writers. Telemetry sometimes retained a previous stage after a final artifact had been written; completion was verified from the scientific outputs and bootstrap state rather than treating an unchanged progress field as a scientific failure.

The Ningbo mitigation run encountered an atomic NPZ writer argument collision: a path parameter named p collided with the prediction-array keyword p. The helper was corrected, a save/reload check verified labels, predictions and thresholds, and validation was recovered from the retained best checkpoint without rerunning completed training. A trainer/finalizer lock handoff and an ECE generator-scoping error were also corrected. These incidents are documented separately from scientific amendments. [S16]

The HEEDB correction was scientifically more consequential because the input length and two lead equations changed the intended experiment. The report retains that failed branch and labels the corrected evaluation honestly. The independent patient confirmation then tested the corrected pipeline on new patients.

On 26 September, a read-only evidence audit independently recomputed aggregate metrics, bootstrap intervals and figure values from saved artifacts, checked six protocol bindings and verified HEEDB cohort separation. It found no discrepancy in the retained final evidence. A separate package check matched all 64 aggregate figure-data rows and figure/source hashes. This verifies arithmetic and provenance; it does not verify clinical code semantics or actual error prevalence. [S12]

## 21 What remains before submission

The remaining work for a submission with the current scientific scope is to write and review the manuscript, finalize authorship contributions, obtain clinical review of the HEEDB outcome crosswalk, check dataset citation/data-use wording and format the paper for a selected venue. The current evidence package already contains aggregate figures and a checked result inventory.

The key clinical questions are whether the core T-wave codes represent the intended positive construct, whether ambiguous related codes should be negative or indeterminate, whether codes 21 and 24 can both represent definite sinus bradycardia, and whether code 23 uniquely supports sinus tachycardia in mixed rhythm statements. The status of physician-overread fields and an exact SNOMED crosswalk also need review. Changing definitions now would create a new exploratory analysis and would not retroactively change the frozen primary result.

For a stronger clinical claim, we would need ECGs with verified actual cable errors, reliable fault provenance, independently reviewed diagnostic outcomes and a defined decision workflow. For a stronger generality claim, we would need a completed additional-model assessment and evaluation across more sites/devices. A prospective comparison with justified equivalence or noninferiority margins would be needed before asserting that our simpler model preserves the comparator's clinical performance.

A new broad method search is not necessary to describe the current result accurately. Any additional experiment should answer one named remaining question and use a separately fixed analysis. No real-fault study, new clinical calibration, physics-aware ECGFounder training or completed ECG-FM evaluation is represented as already done here.

## 22 Code map for mentor review

The project is at AI_research_tests/03_new_biomedical_research/projects/03_physical_ecg_failure_benchmark. The root README contains historical status sections and should not be used alone to infer current scientific completion. The dated final reports and checked submission evidence provide the current scientific status.

| Program or program family | Role in the research |
| --- | --- |
| src/ecg_physics.py | Common terminal permutation and twelve-lead derivation; source of all valid cable transforms. |
| tests/test_ecg_physics.py | Tests electrical identities and exact transformations using synthetic inputs. |
| src/audit_challenge2021_headers.py | Audits public source headers and labels before analysis. |
| src/chapman_stage23.py and src/chapman_v4.py | Public-data ingestion, duplicate handling and canonical split preparation. |
| src/chapman_phase1.py and src/chapman_train_v4.py | Small ResNet definition, raw loader, normalization, loss and initial training. |
| src/chapman_evaluate_v4.py and src/interpret_v5.py | Original missingness evaluation and corrected signed-contrast interpretation. |
| src/matched_reversal_v6.py | Exploratory three-swap and exhaustive five-lead-mask analysis. |
| src/chapman_multiseed_v7.py | Three-seed clean training, checkpoint gate, RA-LL evaluation and bootstrap. |
| src/chapman_multiseed_workflow_v7.py and recovery_v7.py | Supervises and safely resumes the Chapman multiseed workflow. |
| src/ningbo_acquisition_v1.py | Resumable official waveform/header acquisition and checksums. |
| src/ningbo_confirmation_v1.py and v2.py | Ningbo ingestion, split, clean training and confirmation; v2 applies the explicit duplicate amendment. |
| src/mitigation_m1_v1.py and finalize_v1.py | Chapman physics, masking and combined training; saved-score paired bootstrap. |
| src/mitigation_m2_ningbo_v1.py and finalize_v1.py | Locked physics-aware Ningbo follow-up and final analysis. |
| src/paper_readiness_v1.py and post_m2_paper_readiness_v2.py | Aggregate result audits, mechanism summaries and claim boundaries. |
| src/m3_transformer.py and architecture_m3_workflow_v1.py | Additional transformer architecture branch; no completed scientific decision retained. |
| src/ecgfounder_audit_v1.py and preflight_v2.py | Pretrained-weight/label audit, path resolution and missing-sample checks. |
| src/freeze_ecgfounder_baseline_v1.py and v2.py | Pins the separate exploratory Founder protocols and adapters. |
| src/ecgfounder_baseline_v2.py and analyze_v2.py | Frozen feature extraction, diagnostic heads, evaluation and paired bootstrap. |
| src/ecgfounder_report_v2.py and job_supervisor.py | Reports and supervises the completed adapted Founder comparison. |
| src/huang_detector_infer_v1.py and v2.py | Published detector loading, preprocessing and fault-class inference. |
| src/published_detector_pipeline_v1.py | Predicted fault correction followed by saved clean diagnostic ResNets. |
| src/published_detector_multifault_v2.py | Ningbo clean plus three-fault comparison and paired contrasts. |
| src/deployment_tradeoff_v3.py and deployment_latency_*.py | Initial component latency, memory and detector-error diagnostic audit. |
| src/integrated_pipeline_worker_v4.py and integrated_huang_detector_worker_v4.py | Request-response workers for complete diagnostic pipelines. |
| src/integrated_deployment_v4.py | Integrated output-agreement check, warm latency and process-tree memory benchmark. |
| src/heedb_readiness_audit_v1.py and waveform_*_v1.py | Restricted table audit, small readiness sample, fetch and waveform verification. |
| src/freeze_heedb_external_v1.py and heedb_external_v1.py | First frozen external cohort/evaluator; contains the documented invalid implementation. |
| src/heedb_failure_audit_v1.py | Checks labels, units, time axis, algebra and cached outcomes after the first failure. |
| src/heedb_external_corrected_v2.py | Same-cohort correction with exact input shape and common physics library. |
| src/heedb_independent_confirmation_v1.py | New patient-disjoint selection, fixed model transfer and patient bootstrap. |
| src/heedb_confirmation_transfer_amendment_v1a.py | Resumes the final transfer object and verifies completeness without changing science. |
| src/heedb_huang_adapter_v1.py and detector_worker_v1.py | HEEDB-compatible published detector adapter and isolated detector runtime. |
| src/heedb_huang_exploratory_v1.py | Same-patient exploratory correction-chain comparison and detector-error audit. |
| src/submission_evidence_audit_v1.py | Independent arithmetic, protocol, cohort and result verification. |
| src/submission_figures_v1.py and submission_package_check_v1.py | Aggregate figures and deterministic source/value/hash checks. |

The most useful walkthrough begins with ecg_physics.py, the small ResNet in chapman_phase1.py, augmentation in mitigation_m1_v1.py, the Ningbo follow-up, the corrected HEEDB evaluator and the Huang adapter/correction pipeline. The experiment supervisors, downloaders and atomic writers support reliable execution; they are not additional scientific methods.

Frozen protocols reside in configs, experiment artifacts in outputs and restricted HEEDB caches in data/heedb_restricted. The latter are excluded from the shareable report. This report contains aggregate findings, code descriptions and links, without waveform traces, patient identifiers or record-level HEEDB predictions.

## Appendix A Per seed primary results

| Source | Arm | Seed | Clean AP | RA LL AP | Mask AP |
| --- | --- | --- | --- | --- | --- |
| Chapman | Baseline | 104729 | 0.6472 | 0.2753 | 0.5835 |
| Chapman | Baseline | 130363 | 0.6513 | 0.3403 | 0.5535 |
| Chapman | Baseline | 155921 | 0.6609 | 0.3679 | 0.5844 |
| Chapman | Physics | 104729 | 0.6740 | 0.6384 | 0.5754 |
| Chapman | Physics | 130363 | 0.6557 | 0.6287 | 0.5716 |
| Chapman | Physics | 155921 | 0.6625 | 0.6447 | 0.5568 |
| Ningbo | Baseline | 104729 | 0.6736 | 0.1773 | 0.5178 |
| Ningbo | Baseline | 130363 | 0.6756 | 0.1878 | 0.5127 |
| Ningbo | Baseline | 155921 | 0.6791 | 0.1814 | 0.5111 |
| Ningbo | Physics | 104729 | 0.6955 | 0.6741 | 0.5407 |
| Ningbo | Physics | 130363 | 0.6761 | 0.6647 | 0.5191 |
| Ningbo | Physics | 155921 | 0.6890 | 0.6607 | 0.5528 |

The independent HEEDB baseline/physics-aware reversal AUPRC values were 0.1513/0.6040, 0.1829/0.5547 and 0.1599/0.5462 for the same three seed identifiers. The confirmation favors physics-aware ranking for every pair. The three training seeds are not three separately selected patient cohorts.

## Appendix B Scientific stage register

| Scientific branch | Retained status | Meaning |
| --- | --- | --- |
| Phase 0 feasibility | Amended design | Patient-ID requirement blocked original source design. |
| Chapman v3 ingestion | Stopped | Exact duplicate exports identified before training. |
| Chapman v4 and v5 | Mixed or negative | V1-V4 direction did not support hypothesis; signed gate and equivalent masks audited. |
| Reversal development v6 | Exploratory completed | Three swaps characterized; RA-LL selected before Ningbo confirmation. |
| Chapman v7 | Exploratory completed | Three-seed RA-LL gap with exhaustive masks. |
| Ningbo v1 ingestion | Stopped | Three metadata-conflicting duplicate waveform groups. |
| Ningbo v2 baseline | Completed | Six records excluded before split; reversal gap confirmed. |
| Chapman M1 | Completed | Four training setups evaluated; primary physics-aware gate passed. |
| Ningbo M2 | Completed | Locked augmentation follow-up passed; no full four-arm retraining. |
| Transformer M3 | Unfinished | Six Chapman validation units and condition caches retained; no final bootstrap/decision. |
| ECGFounder v1 to v2 | Exploratory completed | Path/NaN adapter resolved; frozen-feature heads evaluated. |
| Huang Ningbo v1 and v2 | Completed comparator | Predicted correction and three faults; diagnostic difference unresolved. |
| Deployment v3 to v4 | Completed benchmark | Integrated end-to-end timing replaced component sums. |
| HEEDB readiness | Completed | Tables and 23 excluded engineering ECGs audited. |
| HEEDB external v1 | Invalid intended test | Wrong temporal length and aVR/aVF transform; artifacts preserved. |
| HEEDB corrected v2 | Post-hoc correction | Same 10,000 patients; strong ranking recovered. |
| HEEDB new confirmation | Completed confirmation | Another 10,000 patients; fixed pipeline result reproduced. |
| HEEDB Huang comparison | Exploratory completed | Planned after new-cohort results; serious comparator, no established AP superiority. |
| Submission evidence | Checked 26 September | Metrics, intervals, cohort separation and three aggregate figures verified. |

This register distinguishes a completed positive finding, a negative/invalid branch, an exploratory comparison and an unfinished branch. A freeze or gate applies to its stated phase; it does not erase previously observed outcomes from the project history.

## Appendix C Evidence files

| ID | Evidence | Project relative path |
| --- | --- | --- |
| S1 | Cross dataset baseline metrics | outputs/paper_readiness_v1/CROSS_DATASET_RESULTS.csv |
| S2 | Chapman ingestion and multiseed decision | outputs/phase1_chapman_seed1_v4/DATA_INGESTION_AUDIT_v4.json |
| S3 | Ningbo ingestion and confirmation | outputs/ningbo_confirmation_v2/FINAL_NINGBO_CONFIRMATION_DECISION.md |
| S4 | Chapman mitigation arms | outputs/mitigation_m1_v1/RESULTS_BY_ARM_SEED.csv |
| S5 | Locked Ningbo mitigation and audit | outputs/post_m2_paper_readiness_v2/FINAL_POST_M2_SCIENTIFIC_AUDIT.md |
| S6 | ECGFounder frozen feature comparison | outputs/ecgfounder_baseline_v2/FINAL_BASELINE_REPORT.md |
| S7 | HEEDB corrected external report | outputs/heedb_external_corrected_v2/FINAL_REPORT.md |
| S8 | HEEDB independent confirmation | outputs/heedb_independent_confirmation_v1/FINAL_REPORT.md |
| S9 | Ningbo published multifault comparator | outputs/published_detector_multifault_v2/FINAL_MULTIFAULT_DECISION.md |
| S10 | HEEDB exploratory Huang comparison | outputs/heedb_huang_exploratory_v1/FINAL_COMPARISON.md |
| S11 | Integrated latency benchmark | outputs/integrated_deployment_v4/FINAL_DECISION.md |
| S12 | Independent scientific evidence check | outputs/submission_evidence_v1/VERIFICATION_REPORT.md |
| S13 | Exact terminal physics implementation | src/ecg_physics.py |
| S14 | Early masking interpretation audit | outputs/phase1_interpretation_audit_v5/FINAL_INTERPRETATION_DECISION.md |
| S15 | Exploratory cable reversal development | outputs/phase1b_matched_reversal_development_v6/FINAL_PHASE1B_DECISION.md |
| S16 | Mitigation recovery and telemetry audit | outputs/post_m2_paper_readiness_v2/INCIDENTS_AND_TELEMETRY_AUDIT.md |
| S17 | HEEDB failed implementation audit | outputs/heedb_failure_audit_v1/FINAL_AUDIT.md |
| S18 | Manuscript readiness and remaining reviews | outputs/submission_evidence_v1/READINESS_DECISION.md |

The source inventory accompanying this report records SHA-256 values for the local evidence files used in compilation. It allows the figures and rounded numbers to be traced to the preserved artifacts without exposing restricted data.

## Appendix D References

[R1] Rudiger A et al. Electrocardiographic artifacts due to electrode misplacement and their frequency in different clinical settings. American Journal of Emergency Medicine, 2007. https://pubmed.ncbi.nlm.nih.gov/17276807/

[R2] Reyna MA et al. Will Two Do Varying Dimensions in Electrocardiography. PhysioNet Computing in Cardiology Challenge 2021, release 1.0.3. Dataset and challenge documentation. https://physionet.org/content/challenge-2021/1.0.3/

[R3] Zheng J et al. A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. Scientific Data, 2020. https://doi.org/10.1038/s41597-020-0386-x

[R4] Koscova Z et al. Harvard Emory ECG Database, version 5.0. BDSP, 2026. https://www.bdsp.io/content/heedb/5.0/ ; dataset DOI https://doi.org/10.60508/rv6h-7d10 ; published dataset paper https://doi.org/10.1038/s41597-026-06861-9 ; repository-requested original preprint https://doi.org/10.1101/2024.09.27.24314503

[R5] Simulating Arbitrary Electrode Reversals in Standard 12-Lead ECG. Sensors, 2019. Electrical transformation background. https://pmc.ncbi.nlm.nih.gov/articles/PMC6651562/

[R6] Li J et al. An Electrocardiogram Foundation Model Built on over 10 Million Recordings. NEJM AI, 2025. https://pubmed.ncbi.nlm.nih.gov/40771651/ ; official code https://github.com/NickLJLee/ECGFounder ; checkpoint https://huggingface.co/PKUDigitalHealth/ECGFounder

[R7] Huang Y et al. A lightweight deep learning approach for detecting electrocardiographic lead misplacement. Physiological Measurement, 2024. https://pubmed.ncbi.nlm.nih.gov/38663434/ ; DOI https://doi.org/10.1088/1361-6579/ad43ae ; code https://github.com/wjcai/ECG_lead_check

[R8] Oh J et al. Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram. CHIL, PMLR 174, 2022. https://proceedings.mlr.press/v174/oh22a.html

[R9] Bo J et al. Pretraining ECG Data with Adversarial Masking Improves Model Generalizability for Data-Scarce Tasks. ML4H, 2022. https://arxiv.org/abs/2211.07889

[R10] Raghu A et al. Data Augmentation for Electrocardiograms. CHIL, PMLR 174, 2022. https://proceedings.mlr.press/v174/raghu22a.html

[R11] Zhu et al. GeoECG Data Augmentation via Wasserstein Geodesic Perturbation for Robust Electrocardiogram Prediction. Machine Learning for Healthcare, PMLR 182, 2022. https://proceedings.mlr.press/v182/zhu22a.html

[R12] Transparent and Robust Artificial Intelligence Driven Electrocardiogram Model for Left Ventricular Systolic Dysfunction. Published study with simulated lead reversal assessment, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12346549/

[R13] Rjoob K et al. Reliable Deep Learning-Based Detection of Misplaced Chest Electrodes During ECG Recording. JMIR Medical Informatics, 2021. https://medinform.jmir.org/2021/4/e25347/

[R14] McKeen K et al. ECG-FM: an open electrocardiogram foundation model. JAMIA Open, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12530324/

Additional representation papers discussed but not implemented here: Gopal et al., 3KG, ML4H 2021, https://proceedings.mlr.press/v158/gopal21a.html ; Kiyasseh et al., CLOCS, ICML 2021, https://proceedings.mlr.press/v139/kiyasseh21a.html ; Lalam et al., ECG Representation Learning with Multi Modal EHR Data, TMLR 2023, https://openreview.net/forum?id=UxmvCwuTMG . They are background references rather than experimental comparisons in this report.

## Appendix E Questions for the mentor

Does the current contribution fit a methods/robustness workshop framing, with the physical-fault benchmark and operational comparison as its core? Can we obtain a clinician review of the HEEDB proxy labels? Should the manuscript retain all secondary outcomes in the supplement while keeping TWA as its main endpoint? Is an additional completed architecture experiment essential for the selected venue, or should it remain a clearly stated limitation? Which team members will take responsibility for manuscript sections, reference checking and final review?

The submission claim I propose is that electrically exact simulated cable reversals expose a diagnostic robustness gap that matched missing-lead tests can miss, and that training on this physical fault family preserves ranking on the evaluated cohorts. The published detector-correction pipeline provides a strong comparator with different operating and computational tradeoffs. This formulation is supported by the current evidence without asserting clinical benefit or SOTA superiority.
