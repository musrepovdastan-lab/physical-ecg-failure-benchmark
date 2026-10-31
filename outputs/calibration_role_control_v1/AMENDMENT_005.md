# Amendment 005 — postprocessing path alias

The supplementary per-fit provenance writer stopped before inference because it referenced an undefined local alias for the Ningbo data root. The failure is recorded in `FAILED_ATTEMPT_005.json`. Existing locked calibration artifacts were not modified by this attempt.

The postprocessor now uses the frozen control module's Ningbo root. It will recreate per-fit configs/provenance and saved model_selection scores, then deterministically recheck A/B scores and the same 36 calibration/threshold entries. The nine fits are not repeated.
