"""Reproducible expression-level synthetic check of project mask evaluators."""
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
MASK = (0, 1, 2, 3, 5)  # I, II, III, aVR, aVF
SOURCE_FILES = (
    "src/ningbo_confirmation_v1.py", "src/mitigation_m2_ningbo_v1.py",
    "src/architecture_m3_workflow_v1.py", "src/mitigation_m1_v1.py",
    "src/chapman_multiseed_workflow_v7.py",
)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> dict:
    # Distinct values per channel and sample; nonzero, nonuniform train stats.
    x = np.arange(12, dtype=np.float64)[:, None] * 10 + np.arange(4)[None, :] + 1
    mean = np.arange(12, dtype=np.float64) * 3 + 101
    std = np.arange(12, dtype=np.float64) * 0.25 + 1.5
    selected = np.isin(np.arange(12), MASK)[:, None]

    # Verbatim branch shape from current evaluators: np.where(mask, x, mean).
    observed_raw = np.where(selected, x, mean[:, None])
    # Correct branch shape from M1/v7: replace selected leads before normalize.
    intended_raw = np.where(selected, mean[:, None], x)
    changed_observed = np.flatnonzero(np.any(observed_raw != x, axis=1)).tolist()
    changed_intended = np.flatnonzero(np.any(intended_raw != x, axis=1)).tolist()
    retained = [i for i in range(12) if i not in MASK]
    sentinel_norm = (mean[list(MASK)] - mean[list(MASK)]) / std[list(MASK)]
    masks = tuple(itertools.combinations(range(12), 5))
    names = tuple("mask5_" + "_".join(map(str, q)) for q in masks)

    result = {
        "check": "PASS",
        "data": "synthetic only; shape=(12 leads, 4 samples); float64",
        "canonical_leads": list(LEADS),
        "train_mean_nonzero": bool(np.all(mean != 0)),
        "train_std_nonzero": bool(np.all(std != 0)),
        "ra_ll_mask_indices": list(MASK),
        "ra_ll_mask_leads": [LEADS[i] for i in MASK],
        "expected_retained_indices": retained,
        "expected_retained_leads": [LEADS[i] for i in retained],
        "inverted_expression_changed_indices": changed_observed,
        "inverted_expression_changed_leads": [LEADS[i] for i in changed_observed],
        "inverted_expression_unchanged_indices": [i for i in range(12) if i not in changed_observed],
        "intended_expression_changed_indices": changed_intended,
        "intended_expression_unchanged_indices": [i for i in range(12) if i not in changed_intended],
        "mean_sentinel_normalizes_to_zero": bool(np.array_equal(sentinel_norm, np.zeros(5))),
        "inverted_selected_signals_preserved": bool(np.array_equal(observed_raw[list(MASK)], x[list(MASK)])),
        "inverted_complement_replaced_by_train_mean": bool(np.array_equal(observed_raw[retained], np.broadcast_to(mean[retained, None], observed_raw[retained].shape))),
        "intended_selected_channels_replaced_by_train_mean": bool(np.array_equal(intended_raw[list(MASK)], np.broadcast_to(mean[MASK, None], intended_raw[list(MASK)].shape))),
        "intended_unmasked_signals_preserved": bool(np.array_equal(intended_raw[retained], x[retained])),
        "panel_mask_count": len(masks),
        "panel_unique_mask_count": len(set(masks)),
        "panel_unique_name_count": len(set(names)),
        "panel_all_have_five_distinct_indices": all(len(q) == 5 and len(set(q)) == 5 for q in masks),
        "source_sha256": {p: sha256(ROOT / p) for p in SOURCE_FILES},
        "source_execution_note": "Expression-level check, not full evaluator invocation. Full evaluator is coupled to model/data loading and inference; its np.where expressions are reproduced above with source files hashed.",
    }
    assert changed_observed == retained
    assert changed_intended == list(MASK)
    assert result["mean_sentinel_normalizes_to_zero"] and result["intended_unmasked_signals_preserved"]
    assert len(masks) == len(set(masks)) == len(set(names)) == 792
    assert result["panel_all_have_five_distinct_indices"]
    out = ROOT / "outputs/mask_semantics_audit_v1/SYNTHETIC_CHECK_RESULTS.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    main()
