# Initialization-seed correction: reviewed and incorporated

The confirmed initialization-order defect in the original calibration-role
control has been addressed in the separate corrected series. In the original
runner, model construction preceded seed setup, so the configured seed did not
determine its initial weights. The corrected run sets Python, NumPy, Torch CPU
and CUDA seeds before constructing the model.

Completed package: [calibration_role_control_seed_corrected_v1](../outputs/calibration_role_control_seed_corrected_v1/).
It contains the frozen protocol, corrected runner, all nine fits, initial-state
and resume audits, A/B calibration and threshold locks, predictions for the
existing Ningbo and HEEDB cohorts, 700 paired bootstrap groups, old/new
comparisons, reports and a hash manifest. All stated computational audits pass.

The original fits remain unchanged historical results. The corrected series is
a reproducibility and sensitivity check on previously examined data, not an
independent external validation. Since the original initial weights are not
known, the old/new difference cannot isolate the causal effect of the seed-order
bug. Initialization matching was tested across methods and fresh processes;
full independent repetitions of all nine trajectories were not performed.

The nine original fits took 70.3 GPU minutes. The corrected fits took 74.70 GPU
minutes on the local RTX 5070 Laptop GPU. Corrected-series inference took about
12.9 minutes across both cohorts; full metrics and statistics took about 5.5
minutes. See the training report for per-run timings and measurement scope.

The follow-up reproduction defects were corrected and independently checked:
the supported input validation passes and the archive writer reproduces exactly
700 canonical groups. All 300 curated local-manifest files matched their hashes;
the nine checkpoints, 96 prediction files and 447 historical artifacts were
verified. No new scientific calculations were needed for the packaging fixes.

The reviewed results are incorporated in
[manuscript version 3](../outputs/submission_evidence_v3/MANUSCRIPT_DRAFT.md),
its [tables](../outputs/submission_evidence_v3/SEED_CORRECTED_RESULTS.md) and
[claim ledger](../outputs/submission_evidence_v3/CLAIMS_AND_EVIDENCE.md).
The remaining work is editorial: author/affiliation details, reference formatting
and venue-specific preparation. The technical seed task is closed; no further
training, new cohort or clinician-contact task is scheduled here.
