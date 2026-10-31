# Open decisions and next steps

Consolidated from the project review dated 20 October 2026. This page summarizes
the remaining decisions; it does not schedule or authorize new experiments.

## Decisions needed

1. **Prior work:** obtain and review the complete Cao 2020 paper and clarify the
   availability of original code and weights. The project review recorded partial
   source access. The proposed choice to omit a direct Cao baseline for the bounded
   evaluation contribution still needs author/mentor agreement.
2. **Clinical interpretation:** obtain clinician feedback on TWA/SB/STach meanings,
   SNOMED and HEEDB proxy mappings, ambiguous/unknown codes, and rules for negative
   labels. The prepared [clinician packet](../outputs/submission_evidence_v2/CLINICIAN_REVIEW_PACKET.md)
   does not constitute clinical sign-off.
3. **Contribution and manuscript:** agree on the limited comparative evaluation
   claim, disclose corrected and negative results, and choose which experiments
   belong in the main paper or supplement. Update the manuscript draft after those
   decisions and the latest controls are incorporated.

## Limitations to retain

- Faults are simulated; real adjudicated miswiring evidence is absent.
- HEEDB endpoint mappings are proxies, and later analyses reuse previously examined data.
- Ningbo record-level splits do not establish patient independence.
- Initialization timing and unrecorded RNG states limit claims about matched
  initialization and training-seed variability. Existing intervals are conditional
  on saved fitted models/calibrators.
- Results depend on cohort, fault mixture and operating point; there is no
  established universal ranking or clinical decision threshold.

## Optional future studies

Leave-one-swap-out evaluation, real miswiring cohorts, additional architectures,
new training seeds and a Cao-inspired adaptation are separate research choices.
They require their own protocols and resources. Previously completed cycle,
mask-recovery and calibration analyses should not be repeated as unfinished work.

For the detailed basis, see [remaining decisions and limitations](../outputs/novelty_positioning_after_controls_v1/REMAINING_DECISIONS_RU.md)
and the [20 October mentor report](../outputs/mentor_update_20261020/REPORT_TEXT_FOR_REVIEW.txt).
