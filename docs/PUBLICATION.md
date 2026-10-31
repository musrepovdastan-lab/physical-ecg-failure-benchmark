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
