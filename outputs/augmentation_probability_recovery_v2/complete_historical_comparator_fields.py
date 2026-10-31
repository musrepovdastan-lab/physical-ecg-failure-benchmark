from __future__ import annotations

import hashlib
from io import BytesIO
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STUDY = ROOT / "outputs/augmentation_probability_study_v1"
sys.path.insert(0, str(ROOT))
from outputs.augmentation_probability_study_v1 import paired_bootstrap as bootstrap_source
from outputs.augmentation_probability_study_v1 import run_probability_study as runner

SEEDS = (104729, 130363, 155921)
HIST_ARMS = ("ordinary", "physics_aware", "combined")
MASKS = tuple(itertools.combinations(range(12), 5))
CONDITIONS = ("clean", "RA_LL_reversal", "cycle_slots_120", "cycle_slots_201", *("mask5_" + "_".join(map(str, x)) for x in MASKS))
ENDPOINTS = ("TWA", "SB", "STach")


def main() -> None:
    supplement = json.loads((OUT / "COMPARATOR_COMPLETENESS_SUPPLEMENT.json").read_text(encoding="utf-8"))
    supplement_hash = supplement.pop("supplement_sha256")
    if hashlib.sha256(json.dumps(supplement, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest() != supplement_hash:
        raise RuntimeError("Historical comparator supplement checksum mismatch")
    supplement["supplement_sha256"] = supplement_hash
    if supplement["parent_recovery_protocol_sha256"] != json.loads((OUT / "RECOVERY_PROTOCOL.json").read_text(encoding="utf-8"))["recovery_protocol_sha256"]:
        raise RuntimeError("Historical comparator supplement has wrong parent recovery protocol")
    source_hashes = supplement["inputs_sha256"]
    for rel, expected in supplement["helper_and_table_sha256"].items():
        source_path = Path(__file__) if rel.endswith("complete_historical_comparator_fields.py") else ROOT / Path(rel)
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Historical comparator completion code changed after supplement freeze")
    recovered = pd.read_csv(OUT / "METRICS_ALL_ARMS_RECOVERED_BY_SEED.csv")
    old = pd.read_csv(STUDY / "METRICS_ALL_ARMS_BY_SEED.csv")
    row_lookup = {
        (r.cohort, r.arm, int(r.seed), r.condition, r.endpoint): int(i)
        for i, r in enumerate(recovered.itertuples(index=False))
    }
    old_keys = ["cohort", "arm", "seed", "condition", "endpoint"]
    old_lookup = {tuple(r[k] for k in old_keys): r for r in old.to_dict("records")}
    order_rows, compare_rows = [], []
    for cohort in ("Chapman", "Ningbo"):
        for arm in HIST_ARMS:
            for seed in SEEDS:
                records, _, _, _, _ = runner.source_data(cohort, seed)
                test_records = records["test"]
                ids = [str(r) if cohort == "Chapman" else str(r["record_id"]) for r in test_records]
                y_expected = runner.labels_for(cohort, test_records).astype(np.uint8)
                order_rows.append({"cohort": cohort, "arm": "physics_p050" if arm == "physics_aware" else arm, "seed": seed, "n_test": len(ids), "canonical_record_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), "labels_sha256": hashlib.sha256(y_expected.tobytes()).hexdigest(), "order_source": "frozen split membership resolved through canonical cohort records"})
                for condition in CONDITIONS:
                    path = bootstrap_source.path_for(cohort, arm, seed, condition)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    relative_path = path.relative_to(ROOT).as_posix()
                    if relative_path not in source_hashes:
                        raise RuntimeError(f"Prediction missing from frozen comparator supplement: {relative_path}")
                    payload = path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != source_hashes[relative_path]:
                        raise RuntimeError(f"Comparator prediction changed after supplement freeze: {relative_path}")
                    with np.load(BytesIO(payload), allow_pickle=False) as z:
                        p = z["p"].astype(np.float64)
                        if "y" in z.files:
                            y = z["y"].astype(np.uint8)
                            if not np.array_equal(y, y_expected):
                                raise RuntimeError(f"historical comparator labels/order mismatch: {cohort}/{arm}/{seed}/{condition}")
                        else:
                            y = y_expected
                    if p.shape != y.shape or p.shape[0] != len(ids) or not np.isfinite(p).all():
                        raise RuntimeError(f"historical prediction shape/nonfinite error: {cohort}/{arm}/{seed}/{condition}")
                    out_arm = "physics_p050" if arm == "physics_aware" else arm
                    for j, endpoint in enumerate(ENDPOINTS):
                        key_old = (cohort, arm, seed, condition, endpoint)
                        base = old_lookup[key_old]
                        key_new = (cohort, out_arm, seed, condition, endpoint)
                        row_i = row_lookup.get(key_new)
                        if row_i is None:
                            raise RuntimeError(f"historical complete-table row missing: {key_new}")
                        yt, pt = y[:, j], p[:, j]
                        threshold = float(base["threshold"])
                        pred = pt >= threshold
                        sens = float(pred[yt == 1].mean())
                        spec = float((~pred[yt == 0]).mean())
                        ppf = float(pred.mean())
                        acc = float((pred == yt).mean())
                        brier = float(np.mean((pt - yt) ** 2))
                        computed = {"sensitivity": sens, "specificity": spec, "predicted_positive_fraction": ppf, "accuracy": acc, "brier": brier}
                        for field, val in computed.items():
                            prior = base.get(field, np.nan)
                            if pd.notna(prior):
                                error = abs(float(prior) - val)
                                compare_rows.append({"cohort": cohort, "arm": out_arm, "seed": seed, "condition": condition, "endpoint": endpoint, "field": field, "v1": float(prior), "from_cached_predictions": val, "absolute_error": error})
                                if error > 1e-6:
                                    raise RuntimeError(f"historical comparator {field} mismatch: {key_new}: {prior} vs {val}")
                            if pd.isna(recovered.at[row_i, field]):
                                recovered.at[row_i, field] = val
    recovered.to_csv(OUT / "METRICS_ALL_ARMS_RECOVERED_BY_SEED.csv", index=False, float_format="%.15g")
    pd.DataFrame(order_rows).to_csv(OUT / "CANONICAL_TEST_ORDER_BINDING_ALL_ARMS.csv", index=False)
    pd.DataFrame(compare_rows).to_csv(OUT / "HISTORICAL_THRESHOLD_METRIC_REPRODUCTION.csv", index=False, float_format="%.15g")
    audit = {
        "historical_arms": ["ordinary", "p050 (source label physics_aware)", "combined"],
        "conditions_per_seed": len(CONDITIONS),
        "cached_predictions_read": len(order_rows) * len(CONDITIONS),
        "inference_performed": False,
        "all_cached_labels_match_canonical_split": True,
        "recomputed_missing_fields": ["Brier where absent", "predicted_positive_fraction where absent", "accuracy where absent"],
        "preexisting_threshold_metrics_checked": len(compare_rows),
        "max_abs_difference_for_preexisting_threshold_metric": max((x["absolute_error"] for x in compare_rows), default=0.0),
        "historical_thresholds_preserved": True,
    }
    (OUT / "HISTORICAL_COMPARATOR_COMPLETENESS_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
