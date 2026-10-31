# Ningbo confirmation v1 readiness audit

**PASS — frozen before Ningbo data access.**

- Official source: PhysioNet Challenge 2021 v1.0.3, `training/ningbo/`, groups g1–g35; expected 34,905 records/patients.
- Official index: per-group `RECORDS`; official checksum manifest: release `SHA256SUMS.txt`.
- Expected acquisition: 69,810 `.hea`/`.mat` source files. Based on the verified Chapman acquisition (~120.8 MB per 1,000 records), download/extracted source storage is estimated at ~4.3 GiB; reserve 25 GiB for data, caches, checkpoints, and logs.
- Current free space: 166.26 GiB (minimum required: 25 GiB).
- Estimated duration: download/checksum 1–3 hours, integrity/split 1–3 hours, three sequential CUDA trainings 2–4 hours, 2,382 condition inferences 1–2 hours, bootstrap 1–3 hours; total approximately 6–15 hours.
- CUDA: project-local environment previously validated with RTX 5070; workflow rechecks it before training.
- Protocol completeness: PASS. It fixes source, checksum verification, duplicate handling, endpoints, seed, split, model, baseline gate, exact fault/comparator, bootstrap, and decision rule.

No Ningbo file, header, label, or waveform was accessed before the protocol hash was recorded.
