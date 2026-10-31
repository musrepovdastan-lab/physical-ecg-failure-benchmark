"""Frozen inference and paired analysis for held-out limb terminal cycles.

Usage from project root:
  .venv/Scripts/python.exe outputs/heldout_limb_cycles_v1/run_heldout_cycles.py freeze
  .venv/Scripts/python.exe outputs/heldout_limb_cycles_v1/run_heldout_cycles.py replay
  .venv/Scripts/python.exe outputs/heldout_limb_cycles_v1/run_heldout_cycles.py infer
  .venv/Scripts/python.exe outputs/heldout_limb_cycles_v1/run_heldout_cycles.py analyze

This script does not train models or modify any historical artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/heldout_limb_cycles_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cycle_adapter import CYCLES, KNOWN_SWAPS, expected_explicit, historical_ra_ll, transform
from src.chapman_phase1 import DATA as CHAPMAN_DATA, ECGResNetSmall, labels as chapman_labels, raw as chapman_raw
from src.chapman_stage23 import parse_header
from src.chest_only_baseline_v1 import ChestOnlyResNetSmall
from src import ningbo_confirmation_v1 as ningbo
from outputs.ningbo_four_arm_replication_v1 import workflow as nw

SEEDS = (104729, 130363, 155921)
ARMS = ("ordinary", "masking", "physics_aware", "combined", "chest_only")
ENDPOINTS = ("TWA", "SB", "STach")
CONDITIONS = ("clean", "RA_LL_reversal", "cycle_slots_120", "cycle_slots_201")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
REPLAY_TOL = 1e-4
BATCH = 128
DEVICE = "cuda"
PROTOCOL = OUT / "FROZEN_PROTOCOL.json"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_bytes(value: bytes):
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def write_json(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def protocol_payload():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected = p.pop("protocol_sha256")
    actual = canonical_hash(p)
    if actual != expected:
        raise RuntimeError(f"frozen protocol hash mismatch: {actual} != {expected}")
    return {**p, "protocol_sha256": expected}


def cohort_records(name):
    if name == "Chapman":
        p = ROOT / "outputs/phase1_chapman_seed1_v4/SPLIT_MEMBERSHIP_v4.json"
        rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
        ids = [r["record_id"] for r in rows if r["split"] == "test"]
        y = np.stack([chapman_labels(r) for r in ids]).astype(np.uint8)
        records = ids
        return records, y, p
    canonical = json.loads((ROOT / "outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json").read_text(encoding="utf-8"))["records"]
    split = json.loads((ROOT / "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json").read_text(encoding="utf-8"))["rows"]
    by = {r["record_id"]: r for r in canonical}
    records = [by[r["record_id"]] for r in split if r["split"] == "test"]
    y = np.asarray([r["labels"] for r in records], dtype=np.uint8)
    return records, y, ROOT / "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json"


def record_ids(name, records):
    return list(records) if name == "Chapman" else [r["record_id"] for r in records]


def model_spec(dataset, arm, seed):
    s = f"seed_{seed}"
    if dataset == "Chapman":
        v4 = ROOT / "outputs/phase1_chapman_seed1_v4"
        v7 = ROOT / "outputs/phase2_chapman_multiseed_v7"
        m1 = ROOT / "outputs/mitigation_m1_v1"
        chest = ROOT / "outputs/chest_only_baseline_v1/models/Chapman" / s
        if arm == "ordinary":
            if seed == 104729:
                d = v4
                return {"checkpoint": d / "ecgresnet_small_seed104729_v4.pt", "preprocessing": d / "preprocessing_v4.npz",
                        "validation": d / "validation_predictions_v4.npz", "cache_dir": d / "test_predictions_v4",
                        "gate": d / "BASELINE_ADEQUACY_v4.json", "gate_type": "baseline_v4", "model_kind": "resnet12"}
            d = v7 / s
            return {"checkpoint": d / "checkpoint.pt", "preprocessing": v4 / "preprocessing_v4.npz", "validation": d / "validation.npz",
                    "cache_dir": d / "test_predictions", "gate": d / "baseline.json", "gate_type": "baseline", "model_kind": "resnet12"}
        if arm == "chest_only":
            return {"checkpoint": chest / "checkpoint.pt", "preprocessing": chest / "preprocessing.npz", "validation": chest / "validation.npz",
                    "cache_dir": chest / "test_predictions", "gate": chest / "training_complete.json", "gate_type": "training", "model_kind": "chest6"}
        d = m1 / arm / s
        return {"checkpoint": d / "checkpoint.pt", "preprocessing": v4 / "preprocessing_v4.npz", "validation": d / "validation.npz",
                "cache_dir": d / "test_predictions", "gate": d / "validation_gate.json", "gate_type": "gate", "model_kind": "resnet12"}

    n = ROOT / "outputs/ningbo_confirmation_v2" / s
    m2 = ROOT / "outputs/mitigation_m2_ningbo_v1" / s
    four = ROOT / "outputs/ningbo_four_arm_replication_v1" / arm / s
    chest = ROOT / "outputs/chest_only_baseline_v1/models/Ningbo" / s
    if arm == "ordinary":
        return {"checkpoint": n / "checkpoint.pt", "preprocessing": n / "preprocessing.npz", "validation": n / "validation.npz",
                "cache_dir": n / "test_predictions", "gate": n / "baseline.json", "gate_type": "baseline", "model_kind": "resnet12"}
    if arm == "physics_aware":
        return {"checkpoint": m2 / "checkpoint.pt", "preprocessing": m2 / "preprocessing.npz", "validation": m2 / "validation.npz",
                "cache_dir": m2 / "test_predictions", "gate": m2 / "validation_gate.json", "gate_type": "gate", "model_kind": "resnet12"}
    if arm == "chest_only":
        return {"checkpoint": chest / "checkpoint.pt", "preprocessing": chest / "preprocessing.npz", "validation": chest / "validation.npz",
                "cache_dir": chest / "test_predictions", "gate": chest / "training_complete.json", "gate_type": "training", "model_kind": "chest6"}
    return {"checkpoint": four / "checkpoint.pt", "preprocessing": four / "preprocessing.npz", "validation": four / "validation.npz",
            "cache_dir": four / "test_predictions", "gate": four / "validation_gate.json", "gate_type": "gate", "model_kind": "resnet12"}


def expected_cache_source(dataset, arm, seed):
    spec = model_spec(dataset, arm, seed)
    if dataset == "Chapman" and arm == "chest_only":
        base = ROOT / "outputs/chest_only_baseline_v1/models/Chapman" / f"seed_{seed}" / "test_predictions"
    elif dataset == "Ningbo" and arm == "chest_only":
        base = ROOT / "outputs/chest_only_baseline_v1/models/Ningbo" / f"seed_{seed}" / "test_predictions"
    else:
        base = spec["cache_dir"]
    return base


def source_paths_for_signal_hash(dataset, records):
    if dataset == "Ningbo":
        p = ROOT / "outputs/ningbo_four_arm_replication_v1/test_waveforms_float32.npy"
        meta = ROOT / "outputs/ningbo_four_arm_replication_v1/TEST_INPUT_CACHE.json"
        return {"kind": "verified_existing_float32_cache", "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha_file(p), "metadata_path": str(meta.relative_to(ROOT)).replace("\\", "/"),
                "metadata_sha256": sha_file(meta)}
    from scipy.io import loadmat
    items = []
    for rid in records:
        base = CHAPMAN_DATA / f"{rid}.mat"
        h = CHAPMAN_DATA / f"{rid}.hea"
        items.append({"record_id": rid, "mat_sha256": sha_file(base), "hea_sha256": sha_file(h)})
    return {"kind": "ordered_source_mat_and_header_hashes", "count": len(items), "ordered_manifest_sha256": canonical_hash(items),
            "records": items}


def cache_path(dataset, arm, seed, condition):
    name = "RA_LL_reversal" if condition == "RA_LL_reversal" else "clean"
    return expected_cache_source(dataset, arm, seed) / f"{name}.npz"


def cache_manifest(dataset, arm, seed, ids_sha=None, labels_sha=None, order_hashes=None, label_hashes=None):
    # Existing completion artifacts bind the source predictions to a checkpoint
    # and, when available, to the ordered test cohort. Preserve the artifact hash.
    sp = model_spec(dataset, arm, seed)
    dirs = [sp["checkpoint"].parent]
    if dataset == "Chapman" and arm == "ordinary" and seed == 104729:
        dirs = [ROOT / "outputs/phase1_chapman_seed1_v4"]
    found = []
    for d in dirs:
        for name in ("evaluation_complete.json", "evaluation_complete_v4.json", "test_predictions_manifest_v4.json"):
            p = d / name
            if p.is_file():
                try:
                    j = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                pred_map = next((j[k] for k in ("prediction_sha256", "prediction_files", "predictions") if isinstance(j.get(k), dict)), {})
                verified_predictions = {}
                for cond in ("clean", "RA_LL_reversal"):
                    expected = pred_map.get(cond)
                    cp = cache_path(dataset, arm, seed, cond)
                    actual = sha_file(cp)
                    if expected is not None and str(expected).lower() != actual.lower():
                        raise RuntimeError(f"historical prediction manifest hash mismatch: {p} => {cond}")
                    verified_predictions[cond] = {"path": str(cp.relative_to(ROOT)).replace("\\", "/"), "sha256": actual,
                                                   "manifest_hash_verified": expected is not None}
                order_value = j.get("test_record_order_sha256", j.get("test_order_sha256", j.get("record_ids_sha256")))
                label_value = j.get("test_labels_sha256")
                valid_order_hashes = {str(v).lower() for v in (order_hashes or [ids_sha]) if v is not None}
                if order_value is not None and valid_order_hashes and str(order_value).lower() not in valid_order_hashes:
                    raise RuntimeError(f"historical cache order binding mismatch: {p}")
                valid_label_hashes = {str(v).lower() for v in (label_hashes or [labels_sha]) if v is not None}
                if label_value is not None and valid_label_hashes and str(label_value).lower() not in valid_label_hashes:
                    raise RuntimeError(f"historical cache label binding mismatch: {p}")
                for cond in ("clean", "RA_LL_reversal"):
                    cp = cache_path(dataset, arm, seed, cond)
                    with np.load(cp, allow_pickle=False) as z:
                        if "binding" in z:
                            binding = json.loads(str(z["binding"].item()))
                            if binding.get("condition") != cond:
                                raise RuntimeError(f"historical cache embedded condition mismatch: {cp}")
                            bound_order = binding.get("test_record_order_sha256")
                            if bound_order is not None and valid_order_hashes and str(bound_order).lower() not in valid_order_hashes:
                                raise RuntimeError(f"historical cache embedded order binding mismatch: {cp}")
                            if binding.get("checkpoint_sha256") and binding["checkpoint_sha256"].lower() != sha_file(sp["checkpoint"]).lower():
                                raise RuntimeError(f"historical cache embedded checkpoint binding mismatch: {cp}")
                found.append({"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(p), "fields": {
                    k: j.get(k) for k in ("checkpoint_sha256", "test_order_sha256", "test_record_order_sha256", "record_ids_sha256", "test_labels_sha256", "conditions", "condition_count", "decision", "seed", "arm") if k in j},
                    "verified_historical_predictions": verified_predictions})
    return found


def load_thresholds(spec):
    with np.load(spec["validation"], allow_pickle=False) as z:
        t = np.asarray(z["thresholds"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.uint8) if "y" in z else None
        p = np.asarray(z["p"], dtype=np.float32) if "p" in z else None
    if t.shape != (3,) or not np.isfinite(t).all():
        raise RuntimeError(f"bad thresholds: {spec['validation']}")
    return t, y, p


def load_state_and_strict(spec):
    obj = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise RuntimeError(f"unexpected checkpoint object: {spec['checkpoint']}")
    sd = obj.get("state_dict", obj)
    if spec["model_kind"] == "chest6":
        model = ChestOnlyResNetSmall()
    else:
        model = ECGResNetSmall()
    model.load_state_dict(sd, strict=True)
    return model, obj


def validate_gate(spec, dataset, arm, seed):
    j = json.loads(spec["gate"].read_text(encoding="utf-8"))
    dec = j.get("decision")
    if dec not in ("PASS", "TRAINED"):
        raise RuntimeError(f"not a passing existing gate: {spec['gate']} => {dec}")
    if j.get("checkpoint_sha256") and j["checkpoint_sha256"].lower() != sha_file(spec["checkpoint"]).lower():
        raise RuntimeError(f"gate checkpoint binding mismatch: {spec['gate']}")
    if "seed" in j and int(j["seed"]) != seed:
        raise RuntimeError(f"seed binding mismatch: {spec['gate']}")
    if "arm" in j and arm not in (j["arm"], "ordinary", "chest_only"):
        raise RuntimeError(f"arm binding mismatch: {spec['gate']}")
    return j


def engineering_checks(sample_by_dataset):
    rng = np.random.default_rng(20261018)
    a = rng.normal(size=(5, 8, 257)).astype(np.float64)
    # Both orderings agree with explicit displayed-lead equations.
    from src.ecg_physics import derive_display, limb_terminal_permutation
    errs = {}
    for cond, p in CYCLES.items():
        got = derive_display(limb_terminal_permutation(a, p))
        disp = derive_display(a)
        expected = expected_explicit(disp, cond)
        errs[cond] = float(np.max(np.abs(got - expected)))
        if errs[cond] > 1e-12:
            raise RuntimeError(f"explicit cycle algebra mismatch: {cond} {errs[cond]}")
    if set(CYCLES.values()) & (set(KNOWN_SWAPS.values()) | {(0, 1, 2)}):
        raise RuntimeError("cycle matches identity or a trained single swap")
    if CYCLES["cycle_slots_120"] != tuple(CYCLES["cycle_slots_201"].index(i) for i in range(3)):
        raise RuntimeError("cycle permutations are not mutual inverses")
    for cond in CYCLES:
        cur = a.copy()
        for _ in range(3):
            cur = limb_terminal_permutation(cur, CYCLES[cond])
        if not np.allclose(cur, a, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"three applications did not restore independent input: {cond}")
    # Explicit two-swap compositions and their application order.
    comp120 = limb_terminal_permutation(limb_terminal_permutation(a, KNOWN_SWAPS["RA_LA_reversal"]), KNOWN_SWAPS["LA_LL_reversal"])
    comp201 = limb_terminal_permutation(limb_terminal_permutation(a, KNOWN_SWAPS["LA_LL_reversal"]), KNOWN_SWAPS["RA_LA_reversal"])
    if not np.allclose(comp120, limb_terminal_permutation(a, CYCLES["cycle_slots_120"]), rtol=0.0, atol=1e-12):
        raise RuntimeError("RA_LA then LA_LL composition does not equal cycle_slots_120")
    if not np.allclose(comp201, limb_terminal_permutation(a, CYCLES["cycle_slots_201"]), rtol=0.0, atol=1e-12):
        raise RuntimeError("LA_LL then RA_LA composition does not equal cycle_slots_201")
    cohort = {}
    for ds, x in sample_by_dataset.items():
        if x.shape[-2:] != (12, 1000):
            raise RuntimeError(f"unexpected {ds} input shape {x.shape}")
        idd = np.asarray(x[:64])
        rederived = derive_display(np.asarray(idd[:, (0, 1, 6, 7, 8, 9, 10, 11), :], dtype=np.float64))
        identity_delta = np.abs(rederived - idd.astype(np.float64))
        cycle_deltas = {c: float(np.max(np.abs(transform(idd, c)[:, 6:] - idd[:, 6:]))) for c in CYCLES}
        if max(cycle_deltas.values()) > 1e-6:
            raise RuntimeError(f"chest leads not invariant within 1e-6: {ds} {cycle_deltas}")
        # Do not change clean. Record actual historical display inconsistency.
        cohort[ds] = {"identity_reconstruction_max_abs_vs_historical_12lead": float(identity_delta.max()),
                      "identity_reconstruction_rmse_vs_historical_12lead": float(np.sqrt(np.mean(identity_delta**2))),
                      "cycle_chest_max_abs_delta": cycle_deltas,
                      "clean_control_policy": "raw historical 12-lead input unchanged; no identity correction"}
    return {"status": "PASS", "synthetic_explicit_formula_max_abs": errs,
            "cycles_distinct_from_identity_and_three_training_swaps": True,
            "cycles_mutual_inverse": True, "triple_application_identity": True,
            "known_swap_compositions": {"cycle_slots_120": "apply RA_LA_reversal then LA_LL_reversal", "cycle_slots_201": "apply LA_LL_reversal then RA_LA_reversal"},
            "batch_channels": list(LEADS), "transform_before_normalization": True,
            "cohort_checks": cohort}


def freeze():
    if PROTOCOL.exists():
        raise RuntimeError("protocol already frozen; use the existing protocol, do not overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = {}
    raw_arrays = {}
    labels_by_dataset = {}
    ids_by_dataset = {}
    for ds in ("Chapman", "Ningbo"):
        records, y, split_path = cohort_records(ds)
        ids = record_ids(ds, records)
        labels_by_dataset[ds] = y
        ids_by_dataset[ds] = ids
        newline_ids_sha = sha_bytes(("\n".join(ids) + "\n").encode("utf-8"))
        if ds == "Ningbo":
            cache = ROOT / "outputs/ningbo_four_arm_replication_v1/test_waveforms_float32.npy"
            meta = json.loads((ROOT / "outputs/ningbo_four_arm_replication_v1/TEST_INPUT_CACHE.json").read_text(encoding="utf-8"))
            x = np.load(cache, mmap_mode="r", allow_pickle=False)
            ids_sha = canonical_hash(ids)
            label_sha = sha_bytes(y.tobytes())
            if list(x.shape) != [len(ids), 12, 1000] or meta.get("test_order_sha256") != ids_sha or meta.get("test_label_order_sha256") != canonical_hash([list(map(int, r)) for r in y]):
                raise RuntimeError("existing Ningbo test input cache order/label binding failed")
            if meta.get("cache_sha256") != sha_file(cache):
                raise RuntimeError("existing Ningbo waveform cache hash mismatch")
            raw_arrays[ds] = x
            signal_binding = source_paths_for_signal_hash(ds, records)
            split_files = [split_path, ROOT / "outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json"]
        else:
            x = np.stack([chapman_raw(r) for r in ids]).astype(np.float32)
            if x.shape != (len(ids), 12, 1000) or not np.isfinite(x).all():
                raise RuntimeError("invalid Chapman test signal array")
            raw_arrays[ds] = x
            ids_sha = canonical_hash(ids)
            newline_ids_sha = sha_bytes(("\n".join(ids) + "\n").encode("utf-8"))
            label_sha = sha_bytes(y.tobytes())
            signal_binding = source_paths_for_signal_hash(ds, ids)
            split_files = [split_path]
        boot = ROOT / ("outputs/phase2_chapman_multiseed_v7/bootstrap_indices.npy" if ds == "Chapman" else "outputs/ningbo_confirmation_v2/bootstrap_indices.npy")
        bix = np.load(boot, mmap_mode="r", allow_pickle=False)
        if bix.shape != (1000, len(ids)) or bix.min() < 0 or bix.max() >= len(ids):
            raise RuntimeError(f"bootstrap shape/index binding invalid: {ds}: {bix.shape}")
        datasets[ds] = {"n": len(ids), "record_order_sha256": ids_sha, "record_order_sha256_newline_join": newline_ids_sha, "label_order_sha256": label_sha,
                        "label_bytes_sha256": sha_bytes(y.tobytes()), "split_and_cohort_files": [{"path": str(q.relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(q)} for q in split_files],
                        "signal_source": signal_binding,
                        "bootstrap": {"path": str(boot.relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(boot), "shape": list(bix.shape), "dtype": str(bix.dtype), "index_min": int(bix.min()), "index_max": int(bix.max()), "resamples": 1000, "record_order_sha256": ids_sha}}

    # Check all 30 immutable models, their gates, validation thresholds, source
    # score files and preprocessing before creating the metric-free protocol.
    units = {}
    for ds in ("Chapman", "Ningbo"):
        n = datasets[ds]["n"]
        ids_sha = datasets[ds]["record_order_sha256"]
        for arm in ARMS:
            for seed in SEEDS:
                spec = model_spec(ds, arm, seed)
                gate = validate_gate(spec, ds, arm, seed)
                model, obj = load_state_and_strict(spec)
                del model
                with np.load(spec["preprocessing"], allow_pickle=False) as z:
                    mean, std = z["mean"], z["std"]
                channels = 6 if spec["model_kind"] == "chest6" else 12
                if mean.shape != (channels,) or std.shape != (channels,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                    raise RuntimeError(f"preprocessing shape/value error: {ds}/{arm}/{seed}")
                thresholds, vy, vp = load_thresholds(spec)
                if vy is not None and (vp is None or vy.ndim != 2 or vy.shape[1] != 3 or vp.shape != vy.shape):
                    raise RuntimeError(f"validation arrays malformed: {ds}/{arm}/{seed}")
                cache_files = {}
                for cond in ("clean", "RA_LL_reversal"):
                    cp = cache_path(ds, arm, seed, cond)
                    with np.load(cp, allow_pickle=False) as z:
                        if "p" not in z or z["p"].shape != (n, 3) or not np.isfinite(z["p"]).all():
                            raise RuntimeError(f"historical cache shape/value error: {cp}")
                        if "y" in z and not np.array_equal(z["y"].astype(np.uint8), labels_by_dataset[ds]):
                            raise RuntimeError(f"historical cache labels differ from frozen order: {cp}")
                        cond_tag = str(z["condition"].item()) if "condition" in z else None
                        if cond_tag and cond_tag != cond:
                            raise RuntimeError(f"historical cache condition metadata mismatch: {cp} {cond_tag}")
                    cache_files[cond] = {"path": str(cp.relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(cp)}
                units[f"{ds}/{arm}/{seed}"] = {
                    "checkpoint": {"path": str(spec["checkpoint"].relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(spec["checkpoint"]),
                                   "state_dict_strict_load": True, "metadata": {k: obj.get(k) for k in ("arm", "seed", "protocol_sha256") if k in obj}},
                    "preprocessing": {"path": str(spec["preprocessing"].relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(spec["preprocessing"]), "channels": channels},
                    "validation_thresholds": {"path": str(spec["validation"].relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(spec["validation"]), "values": thresholds.tolist(), "source": "checkpoint-specific frozen validation thresholds; no new calibration or threshold fit"},
                    "gate": {"path": str(spec["gate"].relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(spec["gate"]), "decision": gate.get("decision")},
                    "historical_score_files": cache_files,
                    "historical_evaluation_manifests": cache_manifest(ds, arm, seed, ids_sha,
                        datasets[ds]["label_bytes_sha256"], [ids_sha, datasets[ds]["record_order_sha256_newline_join"]],
                        [datasets[ds]["label_bytes_sha256"], sha_bytes(labels_by_dataset[ds].astype(np.float32).tobytes())]),
                }

    # The arrays used here are read-only reuses or raw input loads. No new
    # endpoint metric is calculated before the protocol is frozen.
    eng = engineering_checks(raw_arrays)
    source_code = [ROOT / "src/ecg_physics.py", ROOT / "src/mitigation_m1_v1.py", ROOT / "src/mitigation_m2_ningbo_v1.py",
                   ROOT / "src/chapman_multiseed_workflow_v7.py", ROOT / "outputs/ningbo_four_arm_replication_v1/workflow.py",
                   ROOT / "src/chest_only_baseline_v1.py", pathlib.Path(__file__), pathlib.Path(__file__).with_name("cycle_adapter.py")]
    src_h = [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(p)} for p in source_code]
    p = {
        "version": "heldout_limb_cycles_v1", "created_utc": utc(),
        "status": "FROZEN_BEFORE_NEW_METRICS", "scope": "post-access exploratory extension; existing checkpoints only; no new training",
        "conditions": {"cycle_slots_120": {"permutation_slots_RA_LA_LL": [1, 2, 0], "expected_display": ["III", "-I", "-II", "aVL", "aVF", "aVR"]},
                       "cycle_slots_201": {"permutation_slots_RA_LA_LL": [2, 0, 1], "expected_display": ["-II", "-III", "I", "aVF", "aVR", "aVL"]},
                       "chest_leads_invariant": ["V1", "V2", "V3", "V4", "V5", "V6"], "training_faults": ["RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal"],
                       "composition_examples": {"cycle_slots_120": "RA_LA_reversal followed by LA_LL_reversal", "cycle_slots_201": "LA_LL_reversal followed by RA_LA_reversal"}},
        "cohorts": datasets, "models": units, "seeds": list(SEEDS), "arms": list(ARMS), "endpoints": list(ENDPOINTS),
        "conditions_evaluated": list(CONDITIONS), "raw_score_modes": "sigmoid checkpoint outputs; no calibration", "thresholds": "stored per-model validation thresholds; no fitting",
        "inference": {"device": DEVICE, "batch_size": BATCH, "amp": "CUDA float16 autocast", "model_eval": True, "gradients": False,
                      "replay_tolerance_abs_probability": REPLAY_TOL, "replay_indices": "first 256 records in two full batches plus the exact historical final partial batch; compare clean and RA_LL only",
                      "cycle_transforms_before_per_model_normalization": True, "historical_clean_untouched": True},
        "metrics": {"primary": "TWA", "secondary": ["SB", "STach"], "ranking": ["average_precision_score", "roc_auc_score"],
                    "brier": "mean squared probability error", "ece": "15 equal-width bins; empty bins omitted; weighted by bin fraction",
                    "threshold_metrics": ["sensitivity", "specificity", "predicted_positive_fraction"], "bootstrap_replicates": 1000,
                    "bootstrap_aggregation": "same record multiplicity across conditions, arms and seeds; calculate seed-specific metrics then average seeds; no ensemble",
                    "intervals": ["TWA AUPRC", "TWA AUROC", "TWA Brier"], "point_estimate_and_bootstrap_mean_reported_separately": True},
        "comparisons": {"primary": ["physics_aware_minus_ordinary", "combined_minus_ordinary"],
                        "secondary": ["masking_minus_ordinary", "combined_minus_physics_aware", "physics_aware_minus_chest_only", "combined_minus_chest_only"],
                        "conditions": ["each cycle", "clean", "RA_LL_reversal", "cycle_minus_clean", "cycle_minus_RA_LL"]},
        "training_transform_audit": {"result": "cycles absent from generated training examples; each training example receives zero or one transform; sequential composition is not present in the inspected training paths",
                                     "sources": src_h, "no_transform_sequences": True},
        "engineering_checks": eng,
        "analysis_code": {"path": str(pathlib.Path(__file__).relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(__file__),
                          "adapter_path": str(pathlib.Path(__file__).with_name("cycle_adapter.py").relative_to(ROOT)).replace("\\", "/"),
                          "adapter_sha256": sha_file(pathlib.Path(__file__).with_name("cycle_adapter.py"))},
        "limitations": ["historical access to both cohorts", "new cycles are combinations in the existing physical-swap family, not a new fault family", "not real recorded miswirings", "bootstrap intervals condition on fixed models", "cross-zero intervals do not establish equivalence"],
    }
    p["protocol_sha256"] = canonical_hash(p)
    write_json(PROTOCOL, p)
    write_json(OUT / "PROTOCOL_HASH.json", {"protocol_sha256": p["protocol_sha256"], "created_utc": p["created_utc"], "status": p["status"]})
    write_json(OUT / "ENGINEERING_CHECKS.json", eng)
    write_json(OUT / "MODEL_AND_CACHE_AUDIT.json", {"units": units, "cohorts": datasets, "unit_count": len(units), "status": "PASS"})
    print(json.dumps({"status": "FROZEN_BEFORE_NEW_METRICS", "protocol_sha256": p["protocol_sha256"], "model_units": len(units), "datasets": {k: v["n"] for k, v in datasets.items()}, "cycles": list(CYCLES)}, indent=2), flush=True)


def load_raw_array(dataset, records):
    if dataset == "Chapman":
        return np.stack([chapman_raw(r) for r in records]).astype(np.float32)
    x = np.load(ROOT / "outputs/ningbo_four_arm_replication_v1/test_waveforms_float32.npy", mmap_mode="r", allow_pickle=False)
    return x


def normalized_batch(raw, mean, std, arm):
    a = np.asarray(raw, dtype=np.float32)
    if arm == "chest_only":
        a = a[:, 6:12, :]
    return ((a - mean[None, :, None]) / std[None, :, None]).astype(np.float32, copy=False)


def predict_batches(model, raw, mean, std, arm, batch=BATCH):
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(raw), batch):
            x = normalized_batch(raw[start:start + batch], mean, std, arm)
            xt = torch.from_numpy(x).to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = torch.sigmoid(model(xt)).float().cpu().numpy()
            parts.append(out)
    return np.concatenate(parts, axis=0)


def replay():
    p = protocol_payload()
    audit = []
    for ds in ("Chapman", "Ningbo"):
        records, y, _ = cohort_records(ds)
        ids = record_ids(ds, records)
        raw = load_raw_array(ds, records)
        n = len(ids)
        first_ix = np.arange(min(256, n), dtype=np.int64)
        tail_ix = np.arange(n - (n % BATCH), n, dtype=np.int64) if n % BATCH else np.empty(0, dtype=np.int64)
        indices = np.concatenate((first_ix, tail_ix))
        for arm in ARMS:
            for seed in SEEDS:
                ukey = f"{ds}/{arm}/{seed}"
                entry = p["models"][ukey]
                spec = model_spec(ds, arm, seed)
                model, _ = load_state_and_strict(spec)
                model = model.to(DEVICE).eval()
                with np.load(spec["preprocessing"], allow_pickle=False) as z:
                    mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
                ix = indices
                replay_item = {"dataset": ds, "arm": arm, "seed": seed, "n_replayed": len(ix), "replay_indices_sha256": sha_bytes(ix.tobytes()), "tolerance": REPLAY_TOL, "conditions": {}}
                for cond in ("clean", "RA_LL_reversal"):
                    # Match the original inference batch boundaries: two full
                    # batches at the cohort head and the exact historical tail
                    # batch. Never merge a tail into a different full batch.
                    parts = []
                    for chunk_ix in (first_ix, tail_ix):
                        if not len(chunk_ix):
                            continue
                        raw_chunk = np.asarray(raw[chunk_ix])
                        inp = raw_chunk if cond == "clean" else historical_ra_ll(raw_chunk)
                        parts.append(predict_batches(model, inp, mean, std, arm))
                    got = np.concatenate(parts, axis=0)
                    source = cache_path(ds, arm, seed, cond)
                    with np.load(source, allow_pickle=False) as z:
                        ref = np.asarray(z["p"], dtype=np.float32)[ix]
                    delta = np.abs(got - ref)
                    mx = float(delta.max())
                    replay_item["conditions"][cond] = {"source_path": str(source.relative_to(ROOT)).replace("\\", "/"), "source_sha256": sha_file(source), "max_abs_probability_delta": mx,
                                                       "mean_abs_probability_delta": float(delta.mean()), "pass": mx <= REPLAY_TOL}
                replay_item["pass"] = all(x["pass"] for x in replay_item["conditions"].values())
                audit.append(replay_item)
                del model
                torch.cuda.empty_cache()
                print(f"REPLAY {ukey}: {'PASS' if replay_item['pass'] else 'FAIL'}", flush=True)
    write_json(OUT / "REPLAY_AUDIT.json", {"status": "PASS" if all(r["pass"] for r in audit) else "PARTIAL_REPLAY_FAILURE", "tolerance": REPLAY_TOL, "units": audit})
    if not all(r["pass"] for r in audit):
        raise RuntimeError("one or more replay checks failed; affected models must not receive new cycle inference")


def save_cycle_predictions(path, p, y, ids, dataset, arm, seed, cond, unit):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp.npz")
    np.savez_compressed(tmp, p=np.asarray(p, dtype=np.float32), y=np.asarray(y, dtype=np.uint8), record_ids=np.asarray(ids, dtype="U64"),
                        dataset=np.asarray(dataset), arm=np.asarray(arm), seed=np.asarray(seed), condition=np.asarray(cond),
                        checkpoint_sha256=np.asarray(unit["checkpoint"]["sha256"]), preprocessing_sha256=np.asarray(unit["preprocessing"]["sha256"]),
                        protocol_sha256=np.asarray(protocol_payload()["protocol_sha256"]), record_order_sha256=np.asarray(protocol_payload()["cohorts"][dataset]["record_order_sha256"]),
                        threshold_values=np.asarray(unit["validation_thresholds"]["values"], dtype=np.float32))
    os.replace(tmp, path)


def infer():
    p = protocol_payload()
    replay_path = OUT / "REPLAY_AUDIT.json"
    if not replay_path.exists() or json.loads(replay_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("all-model clean/RA_LL replay must pass before cycle inference")
    expected_script = p["analysis_code"]["sha256"]
    if sha_file(__file__) != expected_script or sha_file(pathlib.Path(__file__).with_name("cycle_adapter.py")) != p["analysis_code"]["adapter_sha256"]:
        raise RuntimeError("analysis code changed after protocol freeze; make a protocol amendment before inference")
    done = []
    for ds in ("Chapman", "Ningbo"):
        records, y, _ = cohort_records(ds)
        ids = record_ids(ds, records)
        raw = load_raw_array(ds, records)
        for arm in ARMS:
            for seed in SEEDS:
                ukey = f"{ds}/{arm}/{seed}"
                unit = p["models"][ukey]
                spec = model_spec(ds, arm, seed)
                model, _ = load_state_and_strict(spec)
                model = model.to(DEVICE).eval()
                with np.load(spec["preprocessing"], allow_pickle=False) as z:
                    mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
                unit_out = []
                for cond in CYCLES:
                    outpath = OUT / "predictions" / ds / arm / f"seed_{seed}" / f"{cond}.npz"
                    if outpath.exists():
                        with np.load(outpath, allow_pickle=False) as z:
                            ok = (z["p"].shape == (len(ids), 3) and np.array_equal(z["y"], y) and np.array_equal(z["record_ids"].astype(str), np.asarray(ids).astype(str))
                                  and str(z["protocol_sha256"].item()) == p["protocol_sha256"] and str(z["checkpoint_sha256"].item()) == unit["checkpoint"]["sha256"] and str(z["condition"].item()) == cond)
                        if not ok:
                            raise RuntimeError(f"existing partial cycle prediction has invalid binding: {outpath}")
                        unit_out.append({"condition": cond, "path": str(outpath.relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(outpath), "reused": True})
                        continue
                    chunks = []
                    with torch.inference_mode():
                        for start in range(0, len(raw), BATCH):
                            transformed = transform(np.asarray(raw[start:start + BATCH]), cond)
                            x = normalized_batch(transformed, mean, std, arm)
                            xt = torch.from_numpy(x).to(DEVICE, non_blocking=True)
                            with torch.autocast("cuda", dtype=torch.float16):
                                q = torch.sigmoid(model(xt)).float().cpu().numpy()
                            chunks.append(q)
                    probs = np.concatenate(chunks, axis=0)
                    if probs.shape != (len(ids), 3) or not np.isfinite(probs).all():
                        raise RuntimeError(f"invalid model output {ukey}/{cond}")
                    save_cycle_predictions(outpath, probs, y, ids, ds, arm, seed, cond, unit)
                    unit_out.append({"condition": cond, "path": str(outpath.relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(outpath), "reused": False})
                    print(f"INFER {ukey} {cond}: {len(probs)} records", flush=True)
                done.append({"dataset": ds, "arm": arm, "seed": seed, "status": "COMPLETE", "predictions": unit_out})
                del model
                torch.cuda.empty_cache()
    chest_checks = []
    for ds in ("Chapman", "Ningbo"):
        for seed in SEEDS:
            clean_path = cache_path(ds, "chest_only", seed, "clean")
            with np.load(clean_path, allow_pickle=False) as z:
                clean = np.asarray(z["p"], dtype=np.float32)
            row = {"dataset": ds, "arm": "chest_only", "seed": seed,
                   "clean_cache_path": str(clean_path.relative_to(ROOT)).replace("\\", "/"),
                   "clean_cache_sha256": sha_file(clean_path), "cycles": {}}
            for cond in CYCLES:
                pred_path = OUT / "predictions" / ds / "chest_only" / f"seed_{seed}" / f"{cond}.npz"
                with np.load(pred_path, allow_pickle=False) as z:
                    cycle_p = np.asarray(z["p"], dtype=np.float32)
                delta = np.abs(clean - cycle_p)
                row["cycles"][cond] = {"prediction_path": str(pred_path.relative_to(ROOT)).replace("\\", "/"),
                                       "prediction_sha256": sha_file(pred_path),
                                       "max_abs_probability_delta_vs_clean": float(delta.max()),
                                       "mean_abs_probability_delta_vs_clean": float(delta.mean()),
                                       "exact_probability_match": bool(np.array_equal(clean, cycle_p)),
                                       "pass": bool(delta.max() <= REPLAY_TOL)}
            row["pass"] = all(v["pass"] for v in row["cycles"].values())
            chest_checks.append(row)
    write_json(OUT / "CHEST_ONLY_INVARIANCE_AUDIT.json", {"status": "PASS" if all(r["pass"] for r in chest_checks) else "FAIL",
        "tolerance_abs_probability": REPLAY_TOL, "units": chest_checks})
    if not all(r["pass"] for r in chest_checks):
        raise RuntimeError("complete-cohort chest-only cycle predictions changed beyond replay tolerance")
    write_json(OUT / "INFERENCE_MANIFEST.json", {"status": "COMPLETE" if len(done) == 30 else "PARTIAL", "model_units": done, "protocol_sha256": p["protocol_sha256"], "completed_utc": utc()})


def ece15(y, p):
    bins = np.linspace(0.0, 1.0, 16)
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if np.any(mask):
            out += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return out


def metric_set(y, p, thresholds):
    rows = []
    for j, endpoint in enumerate(ENDPOINTS):
        yy, pp = y[:, j].astype(np.uint8), p[:, j].astype(np.float64)
        pred = pp >= thresholds[j]
        pos, neg = yy == 1, yy == 0
        rows.append({"endpoint": endpoint, "n": int(len(yy)), "positive_count": int(pos.sum()),
                     "auprc": float(average_precision_score(yy, pp)), "auroc": float(roc_auc_score(yy, pp)),
                     "brier": float(np.mean((pp - yy) ** 2)), "ece15": ece15(yy, pp),
                     "sensitivity": float(pred[pos].mean()), "specificity": float((~pred[neg]).mean()),
                     "predicted_positive_fraction": float(pred.mean()), "threshold": float(thresholds[j])})
    return rows


def fast_twa_metrics(y, p):
    """Compute AP/AUROC from one stable score sort, plus Brier.

    Tied scores are accumulated as one threshold group, matching sklearn's
    average_precision_score and roc_auc_score conventions.
    """
    yy = np.asarray(y[:, 0], dtype=np.uint8)
    pp = np.asarray(p[:, 0], dtype=np.float64)
    order = np.argsort(pp, kind="mergesort")
    scores, sorted_y = pp[order], yy[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(scores)]).astype(np.float64)
    positives = np.add.reduceat(sorted_y.astype(np.float64), starts)
    negatives = counts - positives
    total_pos = float(positives.sum())
    total_neg = float(negatives.sum())
    if total_pos <= 0 or total_neg <= 0:
        raise ValueError("ranking metrics require both classes in the sample")

    # Mann-Whitney U with half credit for tied positive-negative pairs.
    neg_before = np.cumsum(negatives) - negatives
    auroc = float(np.sum(positives * (neg_before + 0.5 * negatives)) / (total_pos * total_neg))

    # Average precision at each distinct descending score threshold.
    pos_desc = positives[::-1]
    count_desc = counts[::-1]
    tp = np.cumsum(pos_desc)
    seen = np.cumsum(count_desc)
    auprc = float(np.sum((pos_desc / total_pos) * (tp / seen)))
    brier = float(np.mean((pp - yy) ** 2))
    return {"auprc": auprc, "auroc": auroc, "brier": brier}


def metric_for(y, p, metric):
    try:
        return fast_twa_metrics(y, p)[metric]
    except KeyError as exc:
        raise ValueError(metric) from exc


def load_all_predictions(dataset, records, y):
    p = protocol_payload()
    output = {}
    for arm in ARMS:
        for seed in SEEDS:
            spec = model_spec(dataset, arm, seed)
            with np.load(spec["validation"], allow_pickle=False) as z:
                thresholds = np.asarray(z["thresholds"], dtype=np.float32)
            for cond in CONDITIONS:
                if cond in ("clean", "RA_LL_reversal"):
                    path = cache_path(dataset, arm, seed, cond)
                    with np.load(path, allow_pickle=False) as z: probs = np.asarray(z["p"], dtype=np.float32)
                else:
                    path = OUT / "predictions" / dataset / arm / f"seed_{seed}" / f"{cond}.npz"
                    with np.load(path, allow_pickle=False) as z:
                        probs = np.asarray(z["p"], dtype=np.float32)
                        if not np.array_equal(z["y"], y): raise RuntimeError(f"prediction label mismatch: {path}")
                        if str(z["protocol_sha256"].item()) != p["protocol_sha256"]: raise RuntimeError(f"protocol binding mismatch: {path}")
                if probs.shape != (len(y), 3) or not np.isfinite(probs).all(): raise RuntimeError(f"prediction shape/value mismatch: {dataset}/{arm}/{seed}/{cond}")
                output[(arm, seed, cond)] = (probs, thresholds)
    return output


def make_metric_tables(dataset, ids, y, predictions):
    detailed = []
    for (arm, seed, cond), (prob, thresholds) in predictions.items():
        for m in metric_set(y, prob, thresholds):
            detailed.append({"dataset": dataset, "arm": arm, "seed": seed, "condition": cond, **m})
    summaries = []
    fields = ["auprc", "auroc", "brier", "ece15", "sensitivity", "specificity", "predicted_positive_fraction"]
    groups = {}
    for row in detailed:
        key = (row["dataset"], row["arm"], row["condition"], row["endpoint"])
        groups.setdefault(key, []).append(row)
    for key, rs in groups.items():
        row = dict(zip(("dataset", "arm", "condition", "endpoint"), key))
        row.update({"n": rs[0]["n"], "positive_count": rs[0]["positive_count"]})
        for field in fields:
            values = np.asarray([r[field] for r in rs], dtype=float)
            row[field + "_mean"] = float(values.mean())
            row[field + "_sd"] = float(values.std(ddof=1))
        summaries.append(row)
    return detailed, summaries


def bootstrap_analysis(dataset, y, predictions):
    proto = protocol_payload()
    boot_path = ROOT / proto["cohorts"][dataset]["bootstrap"]["path"]
    if sha_file(boot_path) != proto["cohorts"][dataset]["bootstrap"]["sha256"]:
        raise RuntimeError("bootstrap index file changed after freeze")
    indices = np.load(boot_path, mmap_mode="r", allow_pickle=False)
    nboot = indices.shape[0]
    metrics = ("auprc", "auroc", "brier")
    # For each seed, calculate each condition/model metric on the same sampled rows.
    boot_metric = {}
    for bi in range(nboot):
        ix = np.asarray(indices[bi], dtype=np.int64)
        yy = y[ix]
        for arm in ARMS:
            for seed in SEEDS:
                for cond in CONDITIONS:
                    prob = predictions[(arm, seed, cond)][0][ix]
                    bundle = fast_twa_metrics(yy, prob)
                    for metric in metrics:
                        boot_metric[(bi, arm, seed, cond, metric)] = bundle[metric]
        if (bi + 1) % 100 == 0:
            print(f"BOOTSTRAP {dataset}: {bi+1}/{nboot}", flush=True)

    model_contrasts = [("physics_aware", "ordinary", "physics_aware_minus_ordinary", "primary"),
                       ("combined", "ordinary", "combined_minus_ordinary", "primary"),
                       ("masking", "ordinary", "masking_minus_ordinary", "secondary"),
                       ("combined", "physics_aware", "combined_minus_physics_aware", "secondary"),
                       ("physics_aware", "chest_only", "physics_aware_minus_chest_only", "secondary"),
                       ("combined", "chest_only", "combined_minus_chest_only", "secondary")]
    contrast_draws = []
    summaries = []
    for cycle in CYCLES:
        requests = []
        for left, right, cname, priority in model_contrasts:
            for metric in metrics:
                requests.append((cycle, cname, metric, priority, left, right, cycle, cycle))
        for arm in ARMS:
            for baseline in ("clean", "RA_LL_reversal"):
                label = "cycle_minus_clean" if baseline == "clean" else "cycle_minus_RA_LL"
                for metric in metrics:
                    requests.append((cycle, f"{arm}_{label}", metric, "descriptive", arm, arm, cycle, baseline))
        for cycle_name, cname, metric, priority, left, right, ca, cb in requests:
            draw = np.empty(nboot, dtype=np.float64)
            point_seed_deltas = []
            for seed in SEEDS:
                lprob = predictions[(left, seed, ca)][0]
                rprob = predictions[(right, seed, cb)][0]
                point_seed_deltas.append(metric_for(y, lprob, metric) - metric_for(y, rprob, metric))
            point = float(np.mean(point_seed_deltas))
            for bi in range(nboot):
                seed_delta = [boot_metric[(bi, left, seed, ca, metric)] - boot_metric[(bi, right, seed, cb, metric)] for seed in SEEDS]
                draw[bi] = float(np.mean(seed_delta))
            for bi, value in enumerate(draw):
                contrast_draws.append({"dataset": dataset, "cycle": cycle, "contrast": cname, "metric": metric, "priority": priority, "replicate": bi, "difference": float(value)})
            summaries.append({"dataset": dataset, "cycle": cycle, "contrast": cname, "metric": metric, "priority": priority,
                              "full_cohort_point_estimate": point, "bootstrap_mean": float(draw.mean()), "ci_2_5": float(np.quantile(draw, .025)), "ci_97_5": float(np.quantile(draw, .975)), "replicates": nboot})
    return summaries, contrast_draws


def check_optimized_against_standard(dataset, y, predictions):
    proto = protocol_payload()
    boot_path = ROOT / proto["cohorts"][dataset]["bootstrap"]["path"]
    indices = np.load(boot_path, mmap_mode="r", allow_pickle=False)
    checks = []
    picks = (0, 417, 999)
    for bi in picks:
        ix = np.asarray(indices[bi], dtype=np.int64)
        # The production path metric_for is compared against explicit sklearn and
        # direct Brier calculations on actual repeated records.
        yy = y[ix, 0].astype(np.uint8)
        pp = predictions[("physics_aware", SEEDS[0], "cycle_slots_120")][0][ix, 0].astype(np.float64)
        reference = {"auprc": float(average_precision_score(yy, pp)), "auroc": float(roc_auc_score(yy, pp)), "brier": float(np.mean((pp-yy)**2))}
        actual = {m: metric_for(y[ix], predictions[("physics_aware", SEEDS[0], "cycle_slots_120")][0][ix], m) for m in reference}
        checks.append({"replicate": bi, "max_abs_delta": max(abs(actual[k]-reference[k]) for k in reference), "metrics": actual})
    if any(c["max_abs_delta"] > 1e-12 for c in checks):
        raise RuntimeError("optimized metric implementation disagrees with standard reference")
    return {"status": "PASS", "dataset": dataset, "bootstrap_replicates_compared": list(picks), "checks": checks, "tolerance": 1e-12}


def analyze():
    p = protocol_payload()
    manifest_path = OUT / "INFERENCE_MANIFEST.json"
    if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf-8")).get("status") != "COMPLETE":
        raise RuntimeError("cannot analyze without 30 complete model units")
    all_detailed, all_summary, all_contrasts, all_draws, stat_audits = [], [], [], [], []
    for ds in ("Chapman", "Ningbo"):
        records, y, _ = cohort_records(ds)
        preds = load_all_predictions(ds, records, y)
        detailed, summaries = make_metric_tables(ds, record_ids(ds, records), y, preds)
        all_detailed.extend(detailed); all_summary.extend(summaries)
        bsum, draws = bootstrap_analysis(ds, y, preds)
        all_contrasts.extend(bsum); all_draws.extend(draws)
        stat_audits.append(check_optimized_against_standard(ds, y, preds))
    write_csv(OUT / "METRICS_BY_MODEL_SEED.csv", all_detailed)
    write_csv(OUT / "METRICS_MEAN_SD.csv", all_summary)
    write_csv(OUT / "PAIRED_CONTRASTS_BOOTSTRAP.csv", all_contrasts)
    write_csv(OUT / "BOOTSTRAP_CONTRAST_DRAWS.csv", all_draws)
    write_json(OUT / "STATISTIC_IMPLEMENTATION_AUDIT.json", {"status": "PASS", "checks": stat_audits})
    make_plots(all_summary)
    report(OUT, p, all_summary, all_contrasts)
    hash_manifest()


def make_plots(summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in summaries if r["endpoint"] == "TWA"]
    write_csv(OUT / "PLOT_DATA_TWA.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    colors = {"ordinary": "#777777", "masking": "#d28e00", "physics_aware": "#0072b2", "combined": "#cc4c02", "chest_only": "#009e73"}
    for ax, ds in zip(axes, ("Chapman", "Ningbo")):
        sub = [r for r in rows if r["dataset"] == ds]
        for i, arm in enumerate(ARMS):
            ys = [next(r["auprc_mean"] for r in sub if r["arm"] == arm and r["condition"] == c) for c in CONDITIONS]
            ax.plot(range(4), ys, marker="o", label=arm, color=colors[arm])
        ax.set_title(ds + " (historically accessed test cohort)")
        ax.set_xticks(range(4), ["clean", "RA–LL", "cycle 120", "cycle 201"], rotation=16)
        ax.set_ylabel("TWA AUPRC")
        ax.grid(alpha=.25)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "TWA_AUPRC_BY_CYCLE.png", dpi=180)
    plt.close(fig)


def report(out, p, summaries, contrasts):
    lines = ["# Проверка переноса на две циклические перестановки конечностных электродов", "",
             "**Статус:** расчёты выполнены на существующих моделях; это post-access исследовательское расширение, а не новая независимая подтверждающая проверка. Все цифры относятся к симуляциям перестановки кабелей. Новое обучение не проводилось.", "",
             "## Краткие ответы", "",
             "1. **Помогло ли обучение с перестановками?** См. таблицы ниже: основные сравнения physics-aware и combined против ordinary приведены отдельно для каждого цикла, когорты и seed. Направление/величина эффекта оцениваются по TWA AUPRC и интервалам, без заранее заданного порога успеха.",
             "2. **Сохранение качества.** Для каждого режима показаны clean, знакомая RA–LL и оба цикла, а также парные cycle−clean и cycle−RA–LL различия. Разница цикла с RA–LL сопоставляет разные преобразования; сама по себе она не изолирует «знакомость».",
             "3. **Combined и chest-only.** Они представлены как отдельные frozen режимы. Chest-only дополнительно проверен на совпадение clean и обеих циклических перестановок непосредственно в сохранённых вероятностях.",
             "4. **Chapman против Ningbo.** Направления и величины сравниваются по отдельным когортным таблицам. Обе выборки ранее использовались; совпадение направления не является независимой репликацией.",
             "5. **Граница вывода.** Эксперимент проверяет новые комбинации внутри уже изученного семейства терминальных swaps. Он не доказывает обобщение на иные типы ошибок, реальные зарегистрированные miswirings, клиническую пользу или независимую внешнюю валидность.", "",
             f"Frozen protocol SHA-256: `{p['protocol_sha256']}`. Число моделей: 30; новых prediction-файлов: 60. Пороги — сохранённые per-model validation-пороги. Новая калибровка и threshold fitting не выполнялись.", "",
             "## Метрики по моделям", "",
             "Полные строки для каждого seed и endpoint: `METRICS_BY_MODEL_SEED.csv`. Средние и SD по трём seeds: `METRICS_MEAN_SD.csv`. Основной endpoint — TWA; SB и STach вторичные. Вероятности разных seeds не усреднялись в ансамбль.", "",
             "Для каждого endpoint указаны raw AUPRC/AUROC, Brier, ECE15, sensitivity, specificity и predicted-positive fraction при сохранённом validation threshold. Средние — сначала по seed-specific метрикам; SD рассчитано между тремя seed.", "",
             "| Когорта | Режим | Условие | TWA AUPRC mean ± SD | AUROC mean ± SD | Brier mean ± SD | Sensitivity | Specificity | Predicted-positive fraction |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    tw = [r for r in summaries if r["endpoint"] == "TWA"]
    for r in tw:
        lines.append(f"| {r['dataset']} | {r['arm']} | {r['condition']} | {r['auprc_mean']:.4f} ± {r['auprc_sd']:.4f} | {r['auroc_mean']:.4f} ± {r['auroc_sd']:.4f} | {r['brier_mean']:.4f} ± {r['brier_sd']:.4f} | {r['sensitivity_mean']:.3f} ± {r['sensitivity_sd']:.3f} | {r['specificity_mean']:.3f} ± {r['specificity_sd']:.3f} | {r['predicted_positive_fraction_mean']:.3f} ± {r['predicted_positive_fraction_sd']:.3f} |")
    lines += ["", "## Заранее заданные сравнения и интервалы", "", "Полная когорта и bootstrap mean показаны раздельно. Парные 95% интервалы используют те же 1,000 bootstrap индексов для всех условий, режимов и seeds; в каждом повторе сначала вычисляются seed-specific метрики, затем усредняются три seed. Интервалы условны на фиксированных моделях; пересечение нуля не означает эквивалентность.", "", "| Когорта | Цикл/условие | Контраст | Метрика | Point estimate | Bootstrap mean | 95% CI | Класс |", "|---|---|---|---|---:|---:|---:|---|"]
    for r in contrasts:
        lines.append(f"| {r['dataset']} | {r['cycle']} | {r['contrast']} | {r['metric']} | {r['full_cohort_point_estimate']:+.5f} | {r['bootstrap_mean']:+.5f} | [{r['ci_2_5']:+.5f}, {r['ci_97_5']:+.5f}] | {r['priority']} |")
    lines += ["", "## Инженерная проверка и ограничения", "", "Физические формулы, взаимная обратимость, три применения, композиции известных swaps, инвариантность V1–V6, channel order и identity reconstruction записаны в `ENGINEERING_CHECKS.json`. Исторический clean-вход не корректировался под повторно выведенные отведения; наблюдаемое отличие производных сигналов от реально сохранённых каналов показано числом в инженерном аудите.", "", "Числа являются raw score метриками по ранее определённым proxy labels. Не делалась новая калибровка. Данные и результаты Chapman/Ningbo уже имели предшествующий test access. Сами циклы встречались в литературе; эксперимент не утверждает новизну перестановок.", "", "## Файлы для проверки", "", "- `FROZEN_PROTOCOL.json`, `PROTOCOL_HASH.json` — протокол до новых метрик и его fingerprint.", "- `MODEL_AND_CACHE_AUDIT.json` — 30 checkpoint/preprocessing/gate/cache bindings.", "- `TRAINING_TRANSFORM_AUDIT.md` — происхождение обучающих fault transforms и обоснование unseen-комбинаций.", "- `ENGINEERING_CHECKS.json`, `REPLAY_AUDIT.json` — физические проверки и replay.", "- `INFERENCE_MANIFEST.json`, `predictions/` — 60 новых сохранённых вероятностных файлов с ID/метками/bindings.", "- `METRICS_BY_MODEL_SEED.csv`, `METRICS_MEAN_SD.csv`, `PAIRED_CONTRASTS_BOOTSTRAP.csv`, `BOOTSTRAP_CONTRAST_DRAWS.csv` — анализ и все повторные контрасты.", "- `STATISTIC_IMPLEMENTATION_AUDIT.json` — сверка кода метрик с обычной реализацией на реальных bootstrap выборках.", "- `TWA_AUPRC_BY_CYCLE.png`, `PLOT_DATA_TWA.csv`, `FILE_HASH_MANIFEST.json` — график, использованные агрегаты и хеши.", "", "Этот отчёт не объявляет автоматический критерий успеха. Следующие эксперименты не запускались.", ""]
    (out / "FINAL_HELDOUT_CYCLES_REPORT_RU.md").write_text("\n".join(lines), encoding="utf-8")


def hash_manifest():
    rows = []
    for f in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name != "FILE_HASH_MANIFEST.json"):
        rows.append({"path": f.relative_to(OUT).as_posix(), "bytes": f.stat().st_size, "sha256": sha_file(f)})
    write_json(OUT / "FILE_HASH_MANIFEST.json", {"algorithm": "SHA-256", "files": rows, "self_hash_excluded": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("freeze", "replay", "infer", "analyze"))
    args = ap.parse_args()
    if args.stage == "freeze": freeze()
    elif args.stage == "replay": replay()
    elif args.stage == "infer": infer()
    else: analyze()


if __name__ == "__main__":
    main()
