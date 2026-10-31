# PASS_M2_EXTERNAL_REPLICATION

{
  "decision": "PASS_M2_EXTERNAL_REPLICATION",
  "completed_utc": "2026-08-31T10:30:32.696741+00:00",
  "protocol_sha256": "2D7B605AF175EF6D5B59B223A9EA37BB3F7D675C2C1AC3E27FFF23FCD43E96B8",
  "interpretation": "locked external follow-up replication after observed historical Ningbo baseline outcomes",
  "checks": {
    "ra_ll_twa_improvement_each_seed_min_0_15": true,
    "aggregate_matched_contrast_decrease_min_0_15": true,
    "aggregate_matched_contrast_upper95_max_0_15": true,
    "clean_twa_mean_decline_max_0_02": true,
    "clean_twa_any_seed_decline_max_0_04": true,
    "mean_mask_twa_mean_decline_max_0_02": true,
    "mean_mask_twa_any_seed_decline_max_0_04": true,
    "all_three_seeds_validation_and_integrity": true
  },
  "bootstrap": {
    "TWA_baseline_matched_contrast": {
      "endpoint": "TWA",
      "metric": "baseline_matched_contrast",
      "mean": 0.3311039209365845,
      "low95": 0.3113339841365814,
      "high95": 0.35031962394714355,
      "replicates": 1000,
      "unit": "canonical_record",
      "synchronized": true
    },
    "TWA_m2_matched_contrast": {
      "endpoint": "TWA",
      "metric": "m2_matched_contrast",
      "mean": -0.12873032689094543,
      "low95": -0.1408098191022873,
      "high95": -0.11784562468528748,
      "replicates": 1000,
      "unit": "canonical_record",
      "synchronized": true
    },
    "TWA_contrast_reduction_vs_baseline": {
      "endpoint": "TWA",
      "metric": "contrast_reduction_vs_baseline",
      "mean": 0.4598342478275299,
      "low95": 0.4347999393939972,
      "high95": 0.4838903844356537,
      "replicates": 1000,
      "unit": "canonical_record",
      "synchronized": true
    },
    "SB_baseline_matched_contrast": {
      "endpoint": "SB",
      "metric": "baseline_matched_contrast",
      "mean": 0.037901777774095535,
      "low95": 0.031915370374917984,
      "high95": 0.04489455744624138,
      "replicates": 1000,
      "unit": "canonical_record",
      "synchronized": true
    },
    "SB_m2_matched_contrast": {
      "endpoint": "SB",
      "metric": "m2_matched_contrast",
      "mean": -0.0014475758653134108,
      "low95": -0.0019557976629585028,
      "high95": -0.0010057702893391252,
      "replicates": 1000,
      "unit": "canonical_record",
      "synchronized": true
    },
    "SB_contrast_reduction_vs_baseline": {
      "endpoint": "SB",
      "metric": "contrast_reduction_vs_baseline",
      "mean": 0.03934935852885246,
      "low95": 0.033155716955661774,
      "high95": 0.046634241938591,
      "replicates": 1000,
      "unit": "canonical_record",
      "synchronized": true
    },
    "STach_baseline_matched_contrast": {
      "endpoint": "STach",
      "metric": "baseline_matched_contrast",
      "mean": 0.27143383026123047,
      "low95": 0.24994982779026031,
      "high95": 0.29382193088531494,
      "replicates": 1000,
      "unit": "canonical_record",
      "synchronized": true
    },
    "STach_m2_matched_contrast": {
      "endpoint": "STach",
      "metric": "m2_matched_contrast",
      "mean": -0.007056720554828644,
      "low95": -0.008684623055160046,
      "high95": -0.005292007699608803,
      "replicates": 1000,
      "unit": "canonical_record",
      "synchronized": true
    },
    "STach_contrast_reduction_vs_baseline": {
      "endpoint": "STach",
      "metric": "contrast_reduction_vs_baseline",
      "mean": 0.2784905433654785,
      "low95": 0.25638654828071594,
      "high95": 0.30130594968795776,
      "replicates": 1000,
      "unit": "canonical_record",
      "synchronized": true
    }
  }
}
