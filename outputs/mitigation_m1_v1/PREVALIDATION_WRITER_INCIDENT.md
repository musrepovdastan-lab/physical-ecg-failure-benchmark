# Pre-validation writer incident

At the end of the initial `physics_aware` / seed `104729` training attempt, the
runner raised `TypeError: atomic_npz() got multiple values for argument 'p'` while
writing the validation cache.  The validation-gate JSON was never written and no
test prediction was requested or saved.

The implementation was corrected to use an unambiguous path parameter and to
persist a separate atomic best-model state whenever validation improves.  The
incomplete pre-validation checkpoint and resume state were preserved, without
alteration, under `failed_attempts/physics_aware_seed_104729_prevalidation_writer_error_20260830T112611Z/`.
The only incomplete unit will be rerun from its frozen deterministic seed and
augmentation specification; no completed unit is retrained.

## Evaluation-loader follow-up

The repaired run completed the validation gate and wrote its evaluation freeze,
but before any test batch ran, Windows multiprocessing rejected the function-local
evaluation Dataset as non-picklable.  Evaluation now uses the same frozen
128-record batch size with `num_workers=0`; it does not alter the model, inputs,
fault algebra, masks, thresholds, or any metric.  The gated checkpoint and
validation artifacts are retained and this unit will resume at evaluation only.
