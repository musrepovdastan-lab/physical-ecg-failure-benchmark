# Engineering test report

- Protocol bindings: PASS.
- Existing clean/physics-aware comparator and corrected mask-cache manifest audit: PASS (6 checkpoints; 4,752 corrected mask prediction files; no legacy inverted mask caches reused).
- Synthetic tests of actual transforms/training helpers: PASS; see `engineering_checks_result.json`.
- Validation gates: see `TRAINING_STATUS_BY_ARM_SEED.csv`; test inference is conditional on each unit gate.
- Test set: not accessed by synthetic tests; later test inference follows per-unit validation freeze.
- Frozen method fingerprint: `4718893819c305c0c8efbbe2f9a1e5760d2f622f3f2038fce3685a894acb74ae`.
