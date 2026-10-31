# Engineering test report

Test command: `.venv\Scripts\python.exe -m pytest tests\test_mask_semantics_recovery_v2.py -q`

Result after the final runner changes: **4 passed** in 9.08 seconds.

The tests cover:

- Single-record and batched mask-before-normalization behavior, including unchanged unmasked channels.
- The real inference adapter output shape and finite probabilities.
- Correct five-lead mask enumeration and mask uniqueness.
- Exact weighted-bootstrap AUPRC matching sklearn on tied scores.

Pytest emitted one environment warning because it could not write the workspace `.pytest_cache` node-id file (permission denied). This did not affect test execution or results.
