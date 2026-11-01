# Public snapshot verification

Prepared on 31 October 2026 for mentor review.

- The mathematical test suite passed: 7 tests.
- The full local regression suite passed: 45 tests and 15 subtests; 1 platform-dependent test skipped.
- A clean publication copy, without datasets, checkpoints or vendored ECGFounder source,
  passed 39 tests and 15 subtests; 4 skips cover unavailable upstream code and
  platform support.
- Test temporary files were directed to a writable local folder. No training,
  inference or data acquisition was performed for this publication.
- Selected text files were scanned for common credential patterns. Individual-level
  CSV/JSON exports were excluded, along with all raw/restricted data directories.
- Original experimental artifacts retain their bytes. The only research-test edits make the
  optional upstream ECGFounder dependency skip explicitly when absent; these edits
  do not alter historical protocol hashes or rerun any experiment.

The clean-copy check reused the existing Python 3.12 research environment. It is
not a fresh dependency-install test or a reproduction of the complete experiment
history. The public snapshot is a code/evidence archive with selected aggregate
artifacts; omitted prerequisites are described in [REPRODUCING.md](REPRODUCING.md).

The one-off publication inventory is retained locally. The historical study
manifests and protocols remain alongside the research artifacts; they may include
additional local files that are not distributed.

The repository documentation was subsequently reorganized: methods and protocol
notes are under `docs/`, Phase-0 assessments are under `docs/archive/`, and working
plans are summarized in [next_steps.md](next_steps.md). Research code, tests, frozen
protocols, dataset manifests and experimental results were not changed by that
reorganization.

## Manuscript and corrected-series update — 1 November 2026

Manuscript version 3 incorporates the completed initialization-corrected series
and identifies historical model families separately. The earlier manuscript and
its file-hash manifest remain unchanged. The current abstract, results, tables,
claim ledger, Russian summary and source index use the latest accepted evidence.
The equal-three q=0.1 comparisons against Huang now explicitly report intervals
including zero, while augmentation-versus-ordinary gains remain supported.

Verification for this update:

- Local regression suite: 45 passed, 1 platform-dependent skip; 15 subtests passed.
- Dataset-free publication copy: 39 passed, 4 skips for omitted upstream source
  and platform support; 15 subtests passed.
- Public aggregate verification: 336 TWA seed rows, 112 mean/SD rows and all 700
  paired contrasts checked; maximum point-contrast discrepancy below 1e-10.
- Supported local input validation: original runner/protocol, 65 scientific inputs,
  exact historical documentation snapshots and current documentation provenance pass.
- Original corrected-series scientific artifacts and manuscript v2 retain their
  pre-publication hashes. No additional scientific fits or evaluations were run.

Temporary files were directed to writable project-local folders. The initial
default-temp test runs failed on four temporary-directory access checks; reruns
with explicit TEMP/TMP paths passed without test or research-code changes.

The new/updated public selection is an explicit allowlist in the
[manuscript inventory](../outputs/submission_evidence_v3/PUBLIC_FILE_INVENTORY.json).
Selected text/CSV/JSON files were checked for common credential patterns and
individual-level export fields. Membership contents, waveforms, individual
labels/predictions, model/RNG weights, caches, arrays and external dependencies
remain excluded. The publication copy reused the prepared research environment;
this is not a fresh dependency-install or end-to-end historical reproduction test.

The three current root-document hashes changed explicitly under the
[documentation publication amendment](../outputs/calibration_role_control_seed_corrected_v1/AMENDMENT_004.md).
Original scientific protocol hashes were not replaced.
