# v7 seed stability analysis

{
  "decision": "FREEZE_NINGBO_GENERAL_FAILURE_CONFIRMATION",
  "completed_utc": "2026-08-29T10:51:55.138084+00:00",
  "exploratory": true,
  "ningbo_accessed": false,
  "baseline_all_pass": true,
  "all_seed_twa_contrasts_positive": true,
  "seed_twa_contrasts": {
    "104729": 0.30819098676407153,
    "130363": 0.21319014570812778,
    "155921": 0.2165610782637336
  },
  "bootstrap": {
    "TWA": {
      "label": "TWA",
      "mean_contrast": 0.24571871757507324,
      "low95": 0.20959141850471497,
      "high95": 0.27958938479423523,
      "replicates": 1000,
      "bootstrap_unit": "canonical_record",
      "aggregate": "mean_of_three_seed_contrasts"
    },
    "SB": {
      "label": "SB",
      "mean_contrast": 0.004982891026884317,
      "low95": 0.0030333390459418297,
      "high95": 0.007355509791523218,
      "replicates": 1000,
      "bootstrap_unit": "canonical_record",
      "aggregate": "mean_of_three_seed_contrasts"
    },
    "STach": {
      "label": "STach",
      "mean_contrast": 0.2268005609512329,
      "low95": 0.18671752512454987,
      "high95": 0.2726935148239136,
      "replicates": 1000,
      "bootstrap_unit": "canonical_record",
      "aggregate": "mean_of_three_seed_contrasts"
    }
  },
  "frozen_protocol_sha256": "312DABB9A0400675097D492F66DF58C9DA5EAD69623FFF0563F4DAECE0944CB7"
}
