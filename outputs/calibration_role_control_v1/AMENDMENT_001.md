# Amendment 001 — engineering smoke input encoding

The first pretraining smoke stopped before loading any role waveforms because the accepted membership CSV carries a UTF-8 BOM. The exact failure is recorded in `FAILED_ATTEMPT_001.json`, and the failed freeze is preserved as `FROZEN_EXECUTION_PROTOCOL_FAILED_001.json` with its file hash.

The runner now decodes that CSV as UTF-8 with BOM handling. This is an input parsing correction only: memberships, counts, recipes, model settings, gates, and intended evaluations are unchanged. The execution protocol and runner hashes are regenerated before training. No training, calibration, or external/test scoring occurred before the correction.
