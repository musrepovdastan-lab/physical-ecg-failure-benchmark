# Final post-M2 scientific audit

## Question and intervention

The study asks whether standard ECG classifiers can fail under an electrically exact, physically realizable RA–LL terminal reversal, and whether training on exact terminal reversals reduces that failure. RA–LL reversal swaps the RA and LL terminal connections: displayed leads I, II, III, aVR and aVF change; aVL and V1–V6 are invariant. It is not a missing-channel event. The 792 five-displayed-lead masks are consequence-count matched because five displayed channels change, but they are not waveform-equivalent.

Random masking replaces selected displayed channels with a train-mean sentinel. It therefore tests missing-input tolerance, whereas terminal reversal preserves a coherent signal while changing limb-lead algebra. Physics-aware augmentation uses 50% clean inputs and 50% uniformly sampled exact RA–LA, RA–LL, or LA–LL reversals before normalization.

## Confirmed results

The prespecified Chapman M1 decision was `PASS_M1_FREEZE_NINGBO_M2`; the locked Ningbo M2 external follow-up decision was `PASS_M2_EXTERNAL_REPLICATION`. Both use three seeds, validation gates, 794 conditions per seed, and 1,000 synchronized record-level bootstrap replicates.

For Ningbo TWA, the clean-trained baseline mean RA–LL AUPRC was 0.1822; physics-aware mean RA–LL AUPRC was 0.6665, an absolute improvement of 0.4843. Mean clean AUPRC changed from 0.6761 to 0.6869; mean five-mask AUPRC changed from 0.5139 to 0.5375. The paired matched-contrast reduction was 0.4598 (95% CI [0.4348, 0.4839]).

For Chapman TWA, mean RA–LL AUPRC changed from 0.3278 to 0.6372 (improvement 0.3094); mean clean-performance change was 0.0109 and mean-mask-performance change was -0.0059.

## SB and STach controls

**Measured observation.** On Ningbo, the matched-contrast reduction bootstrap mean was 0.0393 for SB and 0.2785 for STach. Physics-aware models produced near-zero or negative mean matched contrasts for both controls, but this result is endpoint- and model-specific.

**Plausible hypotheses, not established causes.** The clean-trained model may use limb-lead morphology or correlations that are disrupted by reversal even when the clinical label is conceptually linked to rate; augmentation may suppress reliance on such non-invariant cues. Class prevalence, label construction, calibration and score-ranking changes could also contribute.

**Evidence still required.** Feature-attribution or representation analyses, pre-specified subgroup/error audits, alternative architectures, and real acquisition-error data are needed before claiming a clinical mechanism.

## Claim boundary

The evidence confirms robust recovery of this synthetic electrically exact RA–LL reversal failure for the stated architecture, endpoints, splits and augmentation protocol across Chapman and Ningbo. It does not establish performance under real electrode misplacement, patient motion, impedance/open-circuit faults, clinical deployment, or architectures not tested here.
