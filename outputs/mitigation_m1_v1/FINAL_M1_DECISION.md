# PASS_M1_FREEZE_NINGBO_M2

{
  "decision": "PASS_M1_FREEZE_NINGBO_M2",
  "completed_utc": "2026-08-30T19:45:47.610586+00:00",
  "protocol_sha256": "248976F691003DAA45C5F188C53F7CF75891F19F8723223477FC41E1D65667CE",
  "ningbo_accessed": false,
  "primary_arm": "physics_aware",
  "checks": {
    "ra_ll_twa_improvement_each_seed_min_0_15": true,
    "aggregate_matched_contrast_decrease_min_0_15": true,
    "aggregate_matched_contrast_upper95_max_0_15": true,
    "clean_twa_mean_decline_max_0_02": true,
    "clean_twa_any_seed_decline_max_0_04": true,
    "mean_mask_twa_mean_decline_max_0_02": true,
    "mean_mask_twa_any_seed_decline_max_0_04": true,
    "all_physics_seeds_validation_and_integrity": true
  },
  "primary_twa": {
    "baseline_mean_matched_contrast": 0.24598073691197764,
    "physics_mean_matched_contrast": -0.06928439407768627,
    "contrast_reduction": 0.3152651309896639,
    "physics_matched_contrast_bootstrap": {
      "arm": "physics_aware",
      "endpoint": "TWA",
      "metric": "matched_contrast",
      "mean": -0.06899192184209824,
      "low95": -0.0862739086151123,
      "high95": -0.05270916968584061,
      "replicates": 1000,
      "unit": "canonical_record",
      "synchronized_across_arms_seeds_conditions": true
    }
  },
  "secondary_arms": [
    "masking",
    "combined"
  ]
}

M1 is a prespecified Chapman follow-up after observed Chapman outcomes, not a newly sealed confirmation. Ningbo M2 was not started.
