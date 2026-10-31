from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIVATE = ROOT / "data/heedb_restricted/I0006/independent_confirmation_v1"
HUANG_PRIVATE = ROOT / "data/heedb_restricted/I0006/huang_exploratory_v1"
N = ROOT / "outputs/ningbo_confirmation_v2"
M2 = ROOT / "outputs/mitigation_m2_ningbo_v1"
COMBINED = ROOT / "outputs/ningbo_four_arm_replication_v1/combined"
SEEDS = (104729, 130363, 155921)
CONDITIONS = ("clean", "RA_LL_reversal", "RA_LA_reversal", "LA_LL_reversal")
FAULTS = CONDITIONS[1:]
sys.path.insert(0, str(ROOT))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(p: Path, obj: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def verify_protocol() -> dict:
    p = json.loads((OUT / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
    if sha(OUT / "FROZEN_PROTOCOL.json") != (OUT / "FROZEN_PROTOCOL_SHA256.txt").read_text(encoding="utf-8").strip():
        raise RuntimeError("frozen protocol SHA mismatch")
    for rel, binding in p["input_bindings"].items():
        if sha(ROOT / rel) != binding["sha256"]:
            raise RuntimeError("frozen input changed: " + rel)
    return p


def require_torch():
    global torch
    if torch is None:
        import importlib
        torch = importlib.import_module("torch")
    return torch


def reverse_batch(raw: np.ndarray, condition: str) -> np.ndarray:
    if condition == "clean":
        return np.asarray(raw, dtype=np.float32)
    from src.ecg_physics import derive_display, named_reversal
    x = np.asarray(raw, dtype=np.float32)
    independent = np.concatenate((x[:, :2], x[:, 6:]), axis=1)
    return derive_display(named_reversal(independent, condition)).astype(np.float32)


def load_model(family: str, seed: int):
    require_torch()
    from src.chapman_phase1 import ECGResNetSmall
    root = {"ordinary_resnet": N, "physics_aware_p050": M2, "combined": COMBINED}[family]
    d = root / f"seed_{seed}" if family != "combined" else root / f"seed_{seed}"
    ck = d / "checkpoint.pt"
    state = torch.load(ck, map_location="cuda", weights_only=True)
    if int(state.get("seed", -1)) != seed:
        raise RuntimeError(f"checkpoint seed mismatch {family}/{seed}")
    if family == "combined" and state.get("arm") != "combined":
        raise RuntimeError(f"checkpoint arm mismatch {family}/{seed}")
    prep = np.load(d / "preprocessing.npz", allow_pickle=False)
    mean, std = prep["mean"].astype(np.float32), prep["std"].astype(np.float32)
    model = ECGResNetSmall().cuda()
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, mean, std, ck, d / "preprocessing.npz"


def predict(model, raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    require_torch()
    out = []
    with torch.inference_mode():
        for start in range(0, len(raw), 128):
            x = np.asarray(raw[start:start + 128], dtype=np.float32)[:, :, ::5]
            x = ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
            with torch.autocast("cuda", dtype=torch.float16):
                y = torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(x)).cuda())).float().cpu().numpy()
            out.append(y)
    return np.concatenate(out, axis=0)


def predict_100hz(model, raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Ningbo frozen source path is already stride-5 sampled to 100 Hz."""
    require_torch()
    out = []
    with torch.inference_mode():
        for start in range(0, len(raw), 128):
            x = np.asarray(raw[start:start + 128], dtype=np.float32)
            x = ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
            with torch.autocast("cuda", dtype=torch.float16):
                y = torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(x)).cuda())).float().cpu().numpy()
            out.append(y)
    return np.concatenate(out, axis=0)


def replay() -> None:
    verify_protocol()
    from outputs.mixed_fault_calibration_v1 import run_analysis as base
    raw = np.load(PRIVATE / "signals_10000_float32.npy", mmap_mode="r", allow_pickle=False)
    old = np.load(PRIVATE / "predictions_float32.npy", mmap_mode="r", allow_pickle=False)
    rows = []
    refs = (("ordinary_resnet", 0), ("physics_aware_p050", 1))
    for family, arm_ix in refs:
        for si, seed in enumerate(SEEDS):
            model, mean, std, ck, prep = load_model(family, seed)
            for ci, cond in enumerate(("clean", "RA_LL_reversal")):
                batch = reverse_batch(np.asarray(raw[:256]), cond)
                got = predict(model, batch, mean, std)
                ref = np.asarray(old[arm_ix, si, ci, :256], dtype=np.float32)
                delta = float(np.max(np.abs(got - ref)))
                rows.append({"kind": "existing_HEEDB_prediction_replay", "method": family, "seed": seed,
                             "condition": cond, "n": 256, "max_abs_delta": delta, "tolerance": 1e-4,
                             "checkpoint_sha256": sha(ck), "preprocessing_sha256": sha(prep),
                             "status": "PASS" if delta <= 1e-4 else "FAIL"})
            del model
            torch.cuda.empty_cache()
    atomic_json(OUT / "REPLAY_AUDIT.json", {"status": "PASS" if all(x["status"] == "PASS" for x in rows) else "FAIL",
                                           "protocol_sha256": verify_protocol()["protocol_sha256"], "rows": rows,
                                           "sequence": "fixed 256-record HEEDB block after existing Ningbo four-condition cache replay"})
    if not all(x["status"] == "PASS" for x in rows):
        raise RuntimeError("HEEDB replay mismatch; stopped before full cohort inference")
    print(json.dumps({"replay_rows": len(rows), "max_delta": max(x["max_abs_delta"] for x in rows), "status": "PASS"}))


def ningbo_engineering_audit() -> None:
    verify_protocol()
    from src.ecg_physics import derive_display, named_reversal
    from src import published_detector_multifault_v2 as mf
    rows = mf.records()[:256]
    raw = np.stack([mf.raw_display_100hz(r) for r in rows]).astype(np.float32)
    audits = []
    for cond in FAULTS:
        independent = np.concatenate((raw[:, :2], raw[:, 6:]), axis=1)
        physical = derive_display(named_reversal(independent, cond)).astype(np.float32)
        legacy = np.stack([mf.apply_condition(r, cond) for r in raw])
        delta = float(np.max(np.abs(physical - legacy)))
        chest = float(np.max(np.abs(physical[:, 6:] - raw[:, 6:])))
        audits.append({"condition": cond, "records": len(raw), "max_formula_delta_vs_frozen_transform": delta,
                       "max_V1_V6_delta": chest, "status": "PASS" if delta <= 1e-6 and chest <= 1e-6 else "FAIL"})
    # Existing complete cache is the replay target for all four conditions and both source diagnostic arms.
    pred_rows = []
    for method, prefix in (("ordinary_resnet", "diagnostic_clean_resnet_no_correction"),
                           ("physics_aware_p050", "diagnostic_physics_aware_no_correction")):
        for seed in SEEDS:
            model, mean, std, ck, prep = load_model(method, seed)
            for cond in CONDITIONS:
                source = ROOT / f"outputs/published_detector_multifault_v2/cache/{prefix}_{cond}_seed_{seed}.npz"
                if not source.exists():
                    pred_rows.append({"method": method, "seed": seed, "condition": cond, "status": "MISSING"})
                    continue
                refz = np.load(source, allow_pickle=False)
                xb = np.stack([mf.apply_condition(r, cond) for r in raw])
                got = predict_100hz(model, xb, mean, std)
                ref = np.asarray(refz["p"][:256], dtype=np.float32)
                delta = float(np.max(np.abs(got - ref)))
                pred_rows.append({"method": method, "seed": seed, "condition": cond, "n": len(raw),
                                  "max_abs_delta": delta, "tolerance": 1e-4, "status": "PASS" if delta <= 1e-4 else "FAIL"})
            del model
            torch.cuda.empty_cache()
    all_rows = audits + pred_rows
    atomic_json(OUT / "NINGBO_REPLAY_AND_ENGINEERING_AUDIT.json", {"status": "PASS" if all(r["status"] == "PASS" for r in all_rows) else "FAIL",
                                                                     "formula_and_channel_checks": audits,
                                                                     "frozen_prediction_replay": pred_rows,
                                                                     "protocol_sha256": verify_protocol()["protocol_sha256"]})
    if not all(r["status"] == "PASS" for r in all_rows):
        raise RuntimeError("Ningbo replay/engineering gate failed")
    print(json.dumps({"formula_checks": len(audits), "replay_rows": len(pred_rows), "status": "PASS"}))


def infer_diagnostics() -> None:
    verify_protocol()
    audit = json.loads((OUT / "REPLAY_AUDIT.json").read_text(encoding="utf-8"))
    ningbo = json.loads((OUT / "NINGBO_REPLAY_AND_ENGINEERING_AUDIT.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or ningbo["status"] != "PASS":
        raise RuntimeError("replay and engineering gates must pass before full HEEDB inference")
    from outputs.mixed_fault_calibration_v1 import run_analysis as base
    labels, patients = base.heedb_labels()
    selection = __import__("pandas").read_csv(PRIVATE / "selection_initial.csv", dtype=str, keep_default_na=False)
    ids = selection["FileName"].astype(str).to_numpy()
    if len(labels) != 10000 or len(patients) != 10000 or len(ids) != 10000:
        raise RuntimeError("frozen HEEDB labels/IDs length mismatch")
    raw = np.load(PRIVATE / "signals_10000_float32.npy", mmap_mode="r", allow_pickle=False)
    old = np.load(PRIVATE / "predictions_float32.npy", mmap_mode="r", allow_pickle=False)
    method_sources = {"ordinary_resnet": (0, N), "physics_aware_p050": (1, M2), "combined": (None, COMBINED)}
    records = []
    started = time.monotonic()
    for method, (arm_ix, _) in method_sources.items():
        for si, seed in enumerate(SEEDS):
            dest = OUT / "predictions/HEEDB" / method / f"seed_{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            model, mean, std, ck, prep = load_model(method, seed)
            for cond_ix, cond in enumerate(CONDITIONS):
                f = dest / f"{cond}.npz"
                if f.exists():
                    with np.load(f, allow_pickle=False) as z:
                        valid = z["p"].shape == (10000, 3) and np.array_equal(z["labels"], labels) and np.array_equal(z["patient_ids"].astype(str), patients.astype(str)) and str(z["condition"].item()) == cond and str(z["checkpoint_sha256"].item()) == sha(ck)
                    if not valid:
                        raise RuntimeError("invalid existing local prediction " + str(f))
                    records.append({"method": method, "seed": seed, "condition": cond, "status": "REUSED", "path": f.relative_to(ROOT).as_posix(), "sha256": sha(f)})
                    continue
                if method != "combined" and cond_ix < 2:
                    p = np.asarray(old[arm_ix, si, cond_ix], dtype=np.float32)
                    status = "REUSED_FROM_FROZEN_HEEDB_PREDICTIONS"
                elif method == "combined" and cond_ix < 2:
                    p = np.load(ROOT / f"outputs/heedb_four_arm_extension_v1/predictions/combined/seed_{seed}/{cond}.npz", allow_pickle=False)["p"].astype(np.float32)
                    status = "REUSED_FROM_FROZEN_HEEDB_COMBINED"
                else:
                    chunks = []
                    for start in range(0, 10000, 128):
                        batch = reverse_batch(np.asarray(raw[start:min(start + 128, 10000)], dtype=np.float32), cond)
                        chunks.append(predict(model, batch, mean, std))
                        if (start + 128) % 1024 == 0 or start + 128 >= 10000:
                            print(f"INFER {method}/{seed}/{cond} {min(start+128,10000)}/10000", flush=True)
                    p = np.concatenate(chunks, axis=0)
                    status = "NEW_INFERENCE"
                if p.shape != (10000, 3) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    raise RuntimeError("invalid probability array")
                atomic_npz(f, p=p, labels=labels.astype(np.uint8), patient_ids=patients.astype(str), file_names=ids,
                           condition=np.asarray(cond), method=np.asarray(method), seed=np.asarray(seed),
                           checkpoint_sha256=np.asarray(sha(ck)), preprocessing_sha256=np.asarray(sha(prep)),
                           source_ids_sha256=np.asarray(sha(PRIVATE / "selection_initial.csv")),
                           protocol_sha256=np.asarray(verify_protocol()["protocol_sha256"]))
                records.append({"method": method, "seed": seed, "condition": cond, "status": status, "path": f.relative_to(ROOT).as_posix(), "sha256": sha(f)})
            del model
            torch.cuda.empty_cache()
            atomic_json(OUT / "HEEDB_INFERENCE_PROGRESS.json", {"status": "RUNNING", "records": records, "elapsed_seconds": round(time.monotonic()-started, 3)})
    atomic_json(OUT / "HEEDB_DIAGNOSTIC_INFERENCE_COMPLETE.json", {"status": "PASS", "records": records, "elapsed_seconds": round(time.monotonic()-started, 3), "reserved_test_evaluated": False})


def detector_worker() -> None:
    """Run only the already-frozen published four-class detector on the two missing swaps."""
    verify_protocol()
    from src.heedb_huang_adapter_v1 import detector_input, electrical_reversal
    from src.huang_detector_infer_v2 import load_model
    raw = np.load(PRIVATE / "signals_10000_float32.npy", mmap_mode="r", allow_pickle=False)
    model, checkpoint, tf_version = load_model()
    result = {}
    for cond in ("RA_LA_reversal", "LA_LL_reversal"):
        outputs = []
        for start in range(0, 10000, 32):
            stop = min(start + 32, 10000)
            batch = []
            for i in range(start, stop):
                x = np.asarray(raw[i], dtype=np.float32)
                batch.append(detector_input(electrical_reversal(x, cond)))
            p = np.asarray(model.predict(np.asarray(batch, dtype="float32"), verbose=0), dtype=np.float32)
            if p.shape != (stop-start, 4) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-4):
                raise RuntimeError("invalid Huang detector probabilities")
            outputs.append(p)
            if stop % 320 == 0 or stop == 10000: print(f"DETECTOR {cond} {stop}/10000", flush=True)
        result[cond] = np.concatenate(outputs)
    old = np.load(HUANG_PRIVATE / "detector_probabilities_float32.npy", mmap_mode="r", allow_pickle=False)
    stacked = np.stack([old[0], old[1], result["RA_LA_reversal"], result["LA_LL_reversal"]], axis=0)
    atomic_npz(OUT / "huang_detector_probabilities.npz", probabilities=stacked, conditions=np.asarray(CONDITIONS),
               class_names=np.asarray(("clean", "RA_LA", "RA_LL", "LA_LL")),
               source_record_order_sha256=np.asarray(sha(PRIVATE / "selection_initial.csv")),
               detector_index_sha256=np.asarray(sha(Path(checkpoint).with_suffix(".index"))))
    atomic_json(OUT / "HUANG_DETECTOR_INFERENCE_COMPLETE.json", {"status": "PASS", "conditions": CONDITIONS,
               "reused": ["clean", "RA_LL_reversal"], "new": ["RA_LA_reversal", "LA_LL_reversal"],
               "tensorflow_version": tf_version, "checkpoint": str(checkpoint), "protocol_sha256": verify_protocol()["protocol_sha256"]})


def infer_huang_classifier() -> None:
    verify_protocol()
    detector_file = OUT / "huang_detector_probabilities.npz"
    if not detector_file.exists(): raise RuntimeError("Huang detector probabilities missing")
    from src.heedb_huang_adapter_v1 import corrected_batch
    raw = np.load(PRIVATE / "signals_10000_float32.npy", mmap_mode="r", allow_pickle=False)
    d = np.load(detector_file, allow_pickle=False)
    detector_p = d["probabilities"]
    detector_classes = np.argmax(detector_p, axis=-1).astype(np.int8)
    old = np.load(HUANG_PRIVATE / "huang_diagnostic_probabilities_float32.npy", mmap_mode="r", allow_pickle=False)
    rows = []
    for si, seed in enumerate(SEEDS):
        model, mean, std, ck, prep = load_model("ordinary_resnet", seed)
        for ci, cond in enumerate(CONDITIONS):
            f = OUT / "predictions/HEEDB/Huang_full_pipeline" / f"seed_{seed}" / f"{cond}.npz"
            if ci < 2:
                # Existing published detector -> predicted correction -> clean-trained classifier result.
                p = np.asarray(old[si, ci], dtype=np.float32)
                status = "REUSED_FROM_FROZEN_HUANG_PIPELINE"
            else:
                chunks = []
                for start in range(0, 10000, 128):
                    stop = min(start + 128, 10000)
                    corrected = corrected_batch(np.asarray(raw[start:stop], dtype=np.float32), detector_classes[ci, start:stop])
                    chunks.append(predict(model, corrected, mean, std))
                    if stop % 1024 == 0 or stop == 10000: print(f"HUANG {seed}/{cond} {stop}/10000", flush=True)
                p = np.concatenate(chunks, axis=0)
                status = "NEW_FULL_PIPELINE_INFERENCE"
            if p.shape != (10000, 3) or not np.isfinite(p).all(): raise RuntimeError("invalid full Huang pipeline scores")
            atomic_npz(f, p=p, labels=np.load(OUT / "predictions/HEEDB/ordinary_resnet" / f"seed_{seed}/{cond}.npz", allow_pickle=False)["labels"],
                       patient_ids=np.load(OUT / "predictions/HEEDB/ordinary_resnet" / f"seed_{seed}/{cond}.npz", allow_pickle=False)["patient_ids"],
                       condition=np.asarray(cond), method=np.asarray("Huang_full_pipeline"), seed=np.asarray(seed),
                       detector_probabilities_sha256=np.asarray(sha(detector_file)), checkpoint_sha256=np.asarray(sha(ck)),
                       preprocessing_sha256=np.asarray(sha(prep)), protocol_sha256=np.asarray(verify_protocol()["protocol_sha256"]))
            rows.append({"method": "Huang_full_pipeline", "seed": seed, "condition": cond, "status": status, "path": f.relative_to(ROOT).as_posix(), "sha256": sha(f)})
        del model
        torch.cuda.empty_cache()
    atomic_json(OUT / "HEEDB_HUANG_INFERENCE_COMPLETE.json", {"status": "PASS", "rows": rows, "false_correction_clean_included": True,
               "oracle_condition_used": False, "protocol_sha256": verify_protocol()["protocol_sha256"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("replay", "ningbo-audit", "infer-diagnostics", "detector-worker", "infer-huang"))
    args = parser.parse_args()
    {"replay": replay, "ningbo-audit": ningbo_engineering_audit, "infer-diagnostics": infer_diagnostics,
     "detector-worker": detector_worker, "infer-huang": infer_huang_classifier}[args.command]()


if __name__ == "__main__":
    main()
