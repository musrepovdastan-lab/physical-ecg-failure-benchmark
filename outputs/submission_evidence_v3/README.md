# Updated manuscript and aggregate evidence

Version 3, 1 November 2026. This package integrates the completed initialization-corrected calibration-role series into the manuscript. It is a working author-review draft, not a submitted or accepted paper.

- [Manuscript](MANUSCRIPT_DRAFT.md)
- [Corrected-series tables](SEED_CORRECTED_RESULTS.md)
- [Claims and evidence](CLAIMS_AND_EVIDENCE.md)
- [Source index](SOURCE_OF_TRUTH.md)
- [Changes from version 2](CHANGES_FROM_V2.md)
- [Short Russian summary](MENTOR_ONE_PAGE_RU.md)
- [Publication contents and reproduction](PUBLICATION_CONTENTS.md)

The tables are exported from already calculated aggregate results. No new training, inference, calibration, threshold selection or bootstrap generation was performed for this manuscript update. Historical version 2 and scientific artifacts are retained unchanged. The latest series is a sensitivity check on reused Ningbo/HEEDB data, not independent validation.

Run `python outputs/submission_evidence_v3/build_tables.py` to recreate the aggregate tables, and `python outputs/submission_evidence_v3/verify_package.py` to verify this synthesis. Both operate without ECG datasets, checkpoints or individual predictions. Table export uses the published corrected-series aggregate CSVs; see the source bindings and public file inventory.
