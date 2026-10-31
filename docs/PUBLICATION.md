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
- Original file bytes are preserved. The only research-test edits make the
  optional upstream ECGFounder dependency skip explicitly when absent; these edits
  do not alter historical protocol hashes or rerun any experiment.

The clean-copy check reused the existing Python 3.12 research environment. It is
not a fresh dependency-install test or a reproduction of the complete experiment
history. The public snapshot is a code/evidence archive with selected aggregate
artifacts; omitted prerequisites are described in [REPRODUCING.md](REPRODUCING.md).

`PUBLICATION_MANIFEST.json` records distributed file sizes and SHA-256 checksums.
It excludes itself to avoid a circular hash. The local archive and original
historical manifests contain additional artifacts that are not distributed.
