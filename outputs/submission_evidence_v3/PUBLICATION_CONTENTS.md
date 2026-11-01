# Publication contents and reproducibility scope

This manuscript update includes project-owned study code, frozen protocols, audit summaries, human-readable reports and selected aggregate metrics. All numerical exports are derived from completed aggregate result tables. The manuscript is a working draft for author review.

The public corrected-series package omits raw ECGs, labels/predictions at individual level, record membership/split lists, model/optimizer/RNG checkpoints, detector probability caches, resample arrays, private environments, downloaded papers and third-party model/source payloads. The full local manifest describes local artifacts and is not a statement that every listed file is distributed. A separate public inventory identifies the exact distributed files.

The three immutable historical documentation snapshots are included to explain their frozen hashes. Current documentation provenance is updated by an explicit publication amendment, without changing the original scientific protocol or runner bindings. Historical manuscript version 2 is preserved.

On a public clone with pandas and NumPy installed, `build_tables.py` can export this package's tables from the distributed corrected-series aggregate CSVs. `verify_package.py` checks aggregate numerical agreement and the current manuscript manifest. Neither command reads ECG data or individual predictions.

Full research execution needs the separately authorized local prerequisites described in [the reproduction guide](../../docs/REPRODUCING.md) and [supported corrected-series commands](../calibration_role_control_seed_corrected_v1/SUPPORTED_REPRODUCTION.md). The input-only scientific validation requires local artifacts omitted from the public clone; it is not advertised as a clean-clone end-to-end reproduction. Complete independent reruns of all nine full trajectories were not performed for this publication.

The restricted HEEDB data remain subject to their access terms. The project-wide license remains undecided; no new license or permissions for external materials are asserted by this update.
