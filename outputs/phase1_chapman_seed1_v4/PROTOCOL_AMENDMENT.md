# Protocol amendment — Phase 1 Chapman-only pilot v4

## Scope and timing

This pre-outcome amendment supersedes v3 only for exact duplicate-export
handling. It was created after v3's source-integrity audit and before any split,
model training, validation prediction, or Chapman internal-test access. All v3
artifacts remain immutable historical records.

## Frozen policy

All 10,247 official records and 20,494 checksum-verified files remain retained.
An exact duplicate cluster is defined by SHA-256 of the unmodified waveform
matrix/file and requires byte-identical matrix values. A cluster can be collapsed
only when every member also has identical age, sex, and complete diagnosis-code
set. The lexicographically smallest record ID is canonical; other IDs are
duplicate exports, not deleted data. Any conflict yields `STOP_DUPLICATE_CONFLICT`.

The expected result is 13 two-record duplicate-export clusters and 10,234
canonical analytical records. All split, model, metric, and bootstrap operations
will use canonical records only. Source and analytical populations are reported
separately in `DUPLICATE_RESOLUTION.csv` and the v4 ingestion audit.

## Minimal implementation change before outcomes

`src/chapman_v4.py` implements the frozen duplicate ledger and canonical-only
deterministic split. It reuses v3's immutable checksum and duplicate audit
evidence, reads duplicate headers/matrices solely to verify the v4 metadata and
matrix-equality conditions, and does not redownload or alter any source file.

Before execution, a Python quoting syntax defect in this new utility's checksum
comparison was corrected by assigning the existing audit digest to a local
variable before the two comparisons. This was a non-semantic implementation
repair: it did not change any frozen rule, data, endpoint, threshold, or
outcome-handling path.

`src/chapman_train_v4.py` is the minimal v4 output/membership adapter. It imports
the unchanged ECGResNet-small architecture, uses only canonical v4 train and
validation records, and leaves the model seed, optimizer, epochs, loss, mixed
precision, early stopping, and validation gate unchanged.
