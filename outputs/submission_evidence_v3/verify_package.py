"""Verify the public manuscript synthesis using aggregate files only."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((OUT / "FILE_HASH_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = OUT / entry["path"]
        if not path.is_file() or sha(path) != entry["sha256"]:
            raise RuntimeError("manuscript artifact mismatch: " + entry["path"])
    bindings = json.loads((OUT / "TABLE_SOURCE_BINDINGS.json").read_text(encoding="utf-8"))
    for rel, digest in bindings["input_sha256"].items():
        if sha(ROOT / rel) != digest:
            raise RuntimeError("aggregate source changed: " + rel)
    source = ROOT / "outputs/calibration_role_control_seed_corrected_v1"
    pairs = [("SEED_CORRECTED_TWA_MEANS.csv", "MIXTURE_METRICS_MEAN_SD.csv"),
             ("SEED_CORRECTED_TWA_BY_SEED.csv", "MIXTURE_METRICS_BY_SEED.csv")]
    for export, original in pairs:
        a, b = pd.read_csv(OUT / export), pd.read_csv(source / original)
        b = b[b.endpoint == "TWA"].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)
    if sha(OUT / "SEED_CORRECTED_TWA_CONTRASTS.csv") != sha(source / "TWA_PAIRED_CONTRASTS_POINT_AND_CI.csv"):
        raise RuntimeError("paired table export differs from source bytes")
    seeds = pd.read_csv(OUT / "SEED_CORRECTED_TWA_BY_SEED.csv")
    means = pd.read_csv(OUT / "SEED_CORRECTED_TWA_MEANS.csv")
    contrasts = pd.read_csv(OUT / "SEED_CORRECTED_TWA_CONTRASTS.csv")
    assert (len(seeds), len(means), len(contrasts)) == (336, 112, 700)
    qs = {0., .001, .005, .01, .05, .1, 1.}
    assert set(contrasts.q_fault) == qs
    assert set(contrasts.scenario) == {"RA_LL_only", "equal_three_swap"}
    assert (contrasts.n_draws == 1000).all()
    assert contrasts.conditional_on_fitted_models_and_calibrators.all()
    error = 0.
    for row in contrasts.itertuples():
        g = seeds[(seeds.cohort == row.cohort) & (seeds.scenario == row.scenario) &
                  (seeds.q_fault == row.q_fault)]
        left = g[g.method == row.left].set_index("seed")[row.metric]
        right = g[g.method == row.right].set_index("seed")[row.metric]
        assert len(left) == len(right) == 3
        value = float((left - right).mean())
        error = max(error, abs(value - row.full_cohort_point_mean_seed_contrast))
    assert error < 1e-10, error
    current = contrasts[(contrasts.scenario == "equal_three_swap") & (contrasts.q_fault == .1)]
    against_huang = current[(current.right == "Huang_full_pipeline") & (current.metric == "AP_raw")]
    assert len(against_huang) == 4
    assert ((against_huang.ci95_low < 0) & (against_huang.ci95_high > 0)).all()
    against_ordinary = current[(current.right == "ordinary") & (current.metric == "AP_raw") &
                               current.left.isin(["physics_aware_p050", "combined"])]
    assert len(against_ordinary) == 4 and (against_ordinary.ci95_low > 0).all()
    assert np.isfinite(means.select_dtypes(include="number").to_numpy()).all()
    print(json.dumps({"status": "PASS", "manifest_files": len(manifest["files"]),
                      "aggregate_seed_rows": len(seeds), "aggregate_mean_rows": len(means),
                      "paired_groups": len(contrasts), "point_contrast_max_error": error,
                      "q01_huang_comparisons_crossing_zero": len(against_huang),
                      "ecg_data_or_individual_predictions_accessed": False}))


if __name__ == "__main__":
    main()
