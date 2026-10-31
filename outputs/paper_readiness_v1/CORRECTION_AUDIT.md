# Paper-readiness v1 correction audit

This correction changed reporting semantics, lead-algebra explanation, the RA–LL diagram, and the mitigation draft only. No download, model training, inference, 792-mask evaluation, or bootstrap was run.

## Corrected defects

1. Chapman machine-readable fields now use `test_not_accessed_before_evaluation_freeze`; all three source-specific checks pass.
2. Exact RA–LL reversal changes **I, II, III, aVR and aVF**, and leaves **aVL and V1–V6** unchanged. The 792 masks are consequence-matched by count, not waveform-equivalent.

## Scientific invariants

- `CROSS_DATASET_RESULTS.csv` SHA-256 before and after: `b179e2d5d5b41b8eb8f8bd7b63dae4502bd4491a8ba68c7323b752390744443e` (byte-for-byte unchanged).
- Verified result rows: 18; all clean, reversal, mean-mask, contrast and bootstrap fields agree with frozen source CSV/JSON values within the existing CSV serialization precision (1e-9).
- Ningbo remains `CONFIRMED_GENERAL_FAILURE`; all checkpoint and prediction hashes pass.
- Chapman and Ningbo protocol hashes and 1,000-replicate synchronized bootstrap states pass.
