# Where to start

Read the [1 November manuscript](../outputs/submission_evidence_v3/MANUSCRIPT_DRAFT.md),
[corrected-series tables](../outputs/submission_evidence_v3/SEED_CORRECTED_RESULTS.md)
and the [current status](next_steps.md). The
[20 October mentor update](../outputs/mentor_update_20261020/REPORT_TEXT_FOR_REVIEW.txt)
and research-positioning documents are historical context.

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
| Initialization-corrected role-control series | [calibration_role_control_seed_corrected_v1](../outputs/calibration_role_control_seed_corrected_v1/) |
| Held-out limb-cycle compositions | [heldout_limb_cycles_v1](../outputs/heldout_limb_cycles_v1/) |
| Restricted external confirmation: aggregate outputs | [heedb_independent_confirmation_v1](../outputs/heedb_independent_confirmation_v1/) |
| Current manuscript, tables and claim/evidence ledger | [submission_evidence_v3](../outputs/submission_evidence_v3/) |
| Historical manuscript version 2 | [submission_evidence_v2](../outputs/submission_evidence_v2/) |

Manuscript version 3 incorporates the corrected initialization series and the
later cycle/identity sensitivity controls. It is a working author-review draft,
not a submitted paper. Its source index separates current role-control fits from
historical masking, architecture, six-method and runtime results. The original incremental README is preserved
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

## Initialization-seed correction (1 November 2026)

The calibration-role control was rerun in a separate corrected package after
confirming that the old runner constructed the model before setting its seed.
Nine fits (three existing methods × three existing seeds) completed with the
same data roles, training recipes and checkpoint-selection rules. Initialization
tests passed across methods and fresh processes; RNG and loader state were also
verified on an interrupted/resumed miniature trajectory. This establishes
controlled initial weights, but does not independently reproduce all nine full
training trajectories. Old starting weights are unavailable, so old/new changes
are a sensitivity comparison, not an isolated causal estimate of the seed bug.

The high-level result is mixed by cohort: on Ningbo, the point-estimate AP leader
at clean and intermediate fault rates moved from combined to physics-aware p050;
on HEEDB, p050 remained the point leader. Huang retained the point-estimate AP
lead at the fully faulty mixture, though its advantage over p050 on Ningbo is not
supported by the paired interval. At intermediate fault rates, p050 and combined
remained better than ordinary in AP in both cohorts, with intervals excluding
zero. Calibrated Brier scores and false-alarm tradeoffs remain cohort- and
operating-point-dependent. These are conditional comparisons on previously
examined data, not new independent validation.

At equal-three-swap q=0.1, the corrected p050-minus-Huang AP differences are
+0.0126 [−0.0002, +0.0248] on Ningbo and +0.0065 [−0.0056, +0.0174] on HEEDB.
Combined-minus-Huang intervals also include zero in both cohorts. Thus the
historical intermediate-q AP superiority claims against Huang are not sustained
as interval-supported advantages in this repeat. This does not establish
equivalence. Full-precision contrasts and operating tradeoffs are in the
[updated tables](../outputs/submission_evidence_v3/SEED_CORRECTED_RESULTS.md).

The full corrected-series package and its limitations are documented in
[the Russian report](../outputs/calibration_role_control_seed_corrected_v1/FINAL_REPORT_RU.md),
[the per-run training report](../outputs/calibration_role_control_seed_corrected_v1/TRAINING_REPORT_RU.md),
and the [analysis tables and audit files](../outputs/calibration_role_control_seed_corrected_v1/).
