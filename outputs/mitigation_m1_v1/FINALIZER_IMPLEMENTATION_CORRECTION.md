# Finalizer implementation correction

Before final aggregation, `src/mitigation_m1_finalize_v1.py` was corrected so
that sensitivity and specificity use each model's own validation-frozen
thresholds. Cached clean baselines use their historical threshold records; each
mitigation arm/seed uses its own saved `validation.npz` thresholds.

This is a secondary reporting correction only. AUPRC, AUROC, 15-bin ECE,
matched contrasts, all bootstrap AUPRC calculations, and the frozen primary
mitigation decision do not use classification thresholds and therefore cannot
change because of this correction. The corrected finalizer also adds strict
cached-artifact preflight, an atomic resumable bootstrap state, direct-formula
smoke testing, and safe Windows process liveness checks.

The first strict-preflight invocation also caught and immediately corrected an
internal filename comparison that used `.name` rather than `.stem`. All 7,146
mitigation prediction files were present; no bootstrap replicate, inference, or
training ran during that failed check. A successful preflight supersedes that
temporary `STOP_TECHNICAL_BLOCKER` telemetry record before final aggregation.

The first durable launch then stopped before label lookup because its standalone
entry point lacked the project-root import path. This was corrected before the
bootstrap began; no replicate, model load, or inference was performed.
