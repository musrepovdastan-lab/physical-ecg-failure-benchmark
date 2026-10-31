# Engineering amendment 003

The first HEEDB inference invocation stopped before writing any prediction because the NPZ helper used `p` for both the output path parameter and probability-array keyword. Renamed the helper path parameter to `path`; no input, prediction, or statistical procedure changed. The failed attempt is preserved in `FAILED_ATTEMPT_003.json`.
