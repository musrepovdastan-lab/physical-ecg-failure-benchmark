# Where to start

Read the [20 October mentor update](../outputs/mentor_update_20261020/REPORT_TEXT_FOR_REVIEW.txt),
[research positioning](../outputs/novelty_positioning_after_controls_v1/RESEARCH_POSITIONING_RU.md)
and the consolidated [next steps](next_steps.md).

| Question | Study directory |
| --- | --- |
| Corrected mask semantics | [mask_semantics_recovery_v2](../outputs/mask_semantics_recovery_v2/) |
| Corrected augmentation-probability comparisons | [augmentation_probability_recovery_v2](../outputs/augmentation_probability_recovery_v2/) |
| Chest-only explanation | [chest_only_independent_recheck_v1](../outputs/chest_only_independent_recheck_v1/) |
| Architecture replication and its failed full gate | [architecture_replication_m3_recovery_v2](../outputs/architecture_replication_m3_recovery_v2/) |
| Frozen ECGFounder baseline | [ecgfounder_baseline_v2](../outputs/ecgfounder_baseline_v2/) |
| Published detection/correction pipeline | [published_detector_multifault_v2](../outputs/published_detector_multifault_v2/) |
| Four-arm external replication | [ningbo_four_arm_replication_v1](../outputs/ningbo_four_arm_replication_v1/) |
| Identity/rederivation sensitivity | [identity_rederivation_sensitivity_v1](../outputs/identity_rederivation_sensitivity_v1/) |
| Three-swap population mixtures, corrected | [three_swap_population_mixture_recovery_v2](../outputs/three_swap_population_mixture_recovery_v2/) |
| Calibration roles and completed intervals | [calibration_role_control_completion_v2](../outputs/calibration_role_control_completion_v2/) |
| Held-out limb-cycle compositions | [heldout_limb_cycles_v1](../outputs/heldout_limb_cycles_v1/) |
| Restricted external confirmation: aggregate outputs | [heedb_independent_confirmation_v1](../outputs/heedb_independent_confirmation_v1/) |
| Manuscript and claim/evidence ledger | [submission_evidence_v2](../outputs/submission_evidence_v2/) |

The manuscript packet predates the latest controls. It is a coauthor-review draft,
not a final paper. Consult later recovery reports and the 20 October synthesis
before reusing numerical claims. The original incremental README is preserved
in [HISTORY.md](HISTORY.md); its phase-specific status statements are historical.

Historical directories document amendments, stopped runs and superseded analyses.
Follow each study's final report, status and amendment notes. Some historical
links point to intermediates or individual-level files intentionally absent here.

## Interpretation limits

- Faults are simulated; the study does not evaluate real adjudicated miswirings.
- HEEDB outcomes use coded proxy targets. Later extensions reuse previously
  examined data and do not constitute new independent external confirmation.
- Ningbo splits and inference are record-level; patient independence is not established.
- The original calibration-role fits did not seed model initialization. Their
  intervals are conditional on saved models/calibrators; see the [seed correction](next_steps.md).
- Rankings depend on cohort, fault mixture and operating point. Clinical utility
  and a universal decision threshold have not been established.

These statements describe the evidence available in this repository. They do
not add tasks to contact a clinician, acquire new cohorts or collect real faults.
