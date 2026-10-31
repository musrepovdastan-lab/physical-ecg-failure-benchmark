# Engineering gate stop report

**Status:** stopped before head training and before any new test prediction or metric.

## Completed

- The standalone mitigation protocol was frozen before new test outcomes. Its SHA-256 is `8ca374850607497506bb702bed0633fa1a540749c0903043425023a85e85ff4c`.
- The frozen protocol binds 40,624 train/test waveform-header pairs and 42 source/artifact files. Its checksum was independently recomputed and matched.
- The protocol freezes the checkpoint, preprocessing, training/evaluation settings, existing cache replay gates, and all 32 masks.
- The engineering pilot passed. It confirmed a frozen/eval encoder, unchanged encoder state, physical transformations before Founder preprocessing, exact mask behavior, and 1024-dimensional finite features.
- The initial conservative estimate, based on a historical end-to-end throughput of 25.05 records/s with a 1.2 allowance, was 5.83 single-GPU hours.

## Resource gate

The first live extraction run was slower than the historical rate. It stopped at 2,464/7,163 records for Chapman train, `RA_LA_reversal`, after approximately 183 seconds, or about 13.46 records/s. At that measured throughput, the frozen worst-case workload of 437,957 feature rows would take about 9.04 hours. Even assuming the four eligible old test caches (clean and RA_LL for both cohorts) pass their post-freeze replay gate and save 18,062 rows, the remaining 419,895 rows project to about **8.67 single-GPU hours**, above the six-hour limit.

This is an end-to-end single-GPU run-time estimate derived from the live extraction rate; the process used streamed batches and the GPU was not continuously saturated. Because the run cannot be certified under the user-set six-GPU-hour ceiling, dependent training and test evaluation were stopped.

## Preserved state

- `features/chapman_train_clean.npy` and its metadata: verified reuse copy.
- `features/chapman_validation_clean.npy` and its metadata: verified reuse copy.
- `features/chapman_train_RA_LA_reversal.npy` and metadata: resumable `IN_PROGRESS` cache, 2,464 completed records, batch size 16, protocol/checkpoint/order bindings present.
- No head checkpoints, new validation predictions, test predictions, test metrics, or bootstrap results were produced.
- No prior package, model, protocol, or result was modified.

The remaining work would require a revised resource budget or a materially faster verified extraction path. The scientific protocol was not shortened to fit the cap.
