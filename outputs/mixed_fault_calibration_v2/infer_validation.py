"""Replay the frozen Huang adapter, then infer full-pipeline Ningbo validation scores."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "mixed_fault_calibration_v2"
N = ROOT / "outputs" / "ningbo_confirmation_v2"
V1PIPE = ROOT / "outputs" / "published_detector_pipeline_v1"
SEEDS = (104729, 130363, 155921)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_for(split: str) -> list[dict]:
    can = json.loads((N / "CANONICAL_RECORDS.json").read_text(encoding="utf-8"))["records"]
    membership = json.loads((N / "SPLIT_MEMBERSHIP.json").read_text(encoding="utf-8"))["rows"]
    by_id = {r["record_id"]: r for r in can}
    return [by_id[r["record_id"]] for r in membership if r["split"] == split]


def detector_run(split: str, limit: int | None, output: Path) -> dict:
    py = V1PIPE / "detector_env" / "Scripts" / "python.exe"
    cmd = [str(py), str(OUT / "detector_validation_worker.py"), "--split", split, "--output", str(output), "--batch-size", "64"]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    log = OUT / f"detector_{split}{'_replay' if limit else ''}.log"
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=f, stderr=subprocess.STDOUT)
    with np.load(output, allow_pickle=False) as d:
        result = {k: d[k].copy() for k in d.files}
    return result


def diagnostic_scores(rows: list[dict], classes: np.ndarray, seed: int) -> np.ndarray:
    from src.chapman_phase1 import ECGResNetSmall
    from src.published_detector_pipeline_v1 import correct_by_class, load_100hz_display

    prep = np.load(N / f"seed_{seed}" / "preprocessing.npz", allow_pickle=False)
    mean, std = prep["mean"], prep["std"]
    model = ECGResNetSmall().to("cuda")
    checkpoint = torch.load(N / f"seed_{seed}" / "checkpoint.pt", weights_only=True, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    result = []
    batch = []
    with torch.no_grad():
        for i, row in enumerate(rows, 1):
            x = load_100hz_display(row)
            x = correct_by_class(x, int(classes[i - 1]))
            batch.append(((x - mean[:, None]) / std[:, None]).astype("float32"))
            if len(batch) == 128 or i == len(rows):
                xb = torch.from_numpy(np.asarray(batch, dtype="float32")).to("cuda", non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    result.append(torch.sigmoid(model(xb)).float().cpu().numpy())
                batch = []
            if i % 512 == 0 or i == len(rows):
                print(json.dumps({"message": "diagnostic_validation_progress", "seed": seed, "completed": i,
                                  "total": len(rows)}, sort_keys=True), flush=True)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(result).astype("float32", copy=False)


def main() -> None:
    protocol_path = OUT / "FROZEN_COMPLETION_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_VALIDATION_INFERENCE":
        raise RuntimeError("missing frozen pre-inference completion protocol")
    amendment_path = OUT / "FROZEN_COMPLETION_AMENDMENT_001.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8")) if amendment_path.exists() else None
    for rel, binding in protocol["sources"].items():
        if rel == "outputs/mixed_fault_calibration_v2/infer_validation.py" and amendment is not None:
            continue
        path = ROOT / Path(rel)
        if not path.is_file() or sha(path) != binding["sha256"]:
            raise RuntimeError(f"frozen input hash mismatch: {rel}")
    if amendment is not None:
        if amendment.get("status") != "FROZEN_BEFORE_REPLAY_RETRY" or amendment.get("protocol_sha256") != sha(protocol_path):
            raise RuntimeError("invalid frozen replay amendment")
        if amendment.get("infer_validation_py_sha256") != sha(Path(__file__)):
            raise RuntimeError("replay amendment does not bind current inference adapter")
    start = time.monotonic()
    val_rows = rows_for("validation")
    test_rows = rows_for("test")
    if len(val_rows) != 3488 or len(test_rows) != 6981:
        raise RuntimeError("frozen population count changed")
    detector_replay = detector_run("test", 64, OUT / "detector_test_replay.npz")
    prior_detector = np.load(V1PIPE / "cache/detector_test_probs.npz", allow_pickle=False)
    prior_ids = prior_detector["record_ids"].astype(str)
    target_ids = np.asarray([r["record_id"] for r in test_rows[:64]])
    old_detector_p = prior_detector["probs_clean"][:64]
    if not np.array_equal(detector_replay["record_ids"].astype(str), target_ids):
        raise RuntimeError("detector replay record order differs from historical pipeline")
    detector_delta = float(np.max(np.abs(detector_replay["probabilities"] - old_detector_p)))
    detector_tol = protocol["pipeline"]["replay_tolerances"]["detector_max_abs_probability"]
    if detector_delta > detector_tol:
        raise RuntimeError(f"detector replay mismatch {detector_delta} > frozen tolerance {detector_tol}")
    # The historical clean pipeline's predicted class controls correction for every clean ECG.
    from src.published_detector_pipeline_v1 import records as old_test_records
    old_test = old_test_records()
    old_test_classes = prior_detector["probs_clean"].argmax(axis=1)
    validation_detector = detector_run("validation", None, OUT / "detector_validation.npz")
    val_ids = np.asarray([r["record_id"] for r in val_rows])
    if not np.array_equal(validation_detector["record_ids"].astype(str), val_ids):
        raise RuntimeError("validation detector output record order mismatch")
    if validation_detector["probabilities"].shape != (3488, 4) or not np.isfinite(validation_detector["probabilities"]).all():
        raise RuntimeError("invalid validation detector score shape or nonfinite values")
    if not np.allclose(validation_detector["probabilities"].sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("detector probabilities fail simplex check")
    val_classes = validation_detector["probabilities"].argmax(axis=1)
    clean_incorrect = float(np.mean(val_classes != 0))
    diag_replay = {}
    for seed in SEEDS:
        prior = np.load(V1PIPE / f"cache/huang_corrected_clean_seed_{seed}.npz", allow_pickle=False)["p"]
        # Historical outputs were produced in full batches of 128. Replaying 128
        # preserves the same GPU kernel shape; the comparison remains the first
        # 64 preregistered records and the frozen numeric tolerance is unchanged.
        new = diagnostic_scores(old_test[:128], old_test_classes[:128], seed)
        delta = float(np.max(np.abs(new[:64] - prior[:64])))
        tol = protocol["pipeline"]["replay_tolerances"]["diagnostic_max_abs_probability"]
        if delta > tol:
            raise RuntimeError(f"diagnostic replay mismatch seed={seed}: {delta} > {tol}")
        diag_replay[str(seed)] = {"max_abs_probability_delta": delta, "tolerance": tol, "compared_n": 64,
                                  "inference_batch_size": 128, "inferred_n": len(new)}
    diagnostic = []
    for seed in SEEDS:
        diagnostic.append(diagnostic_scores(val_rows, val_classes, seed))
    score = np.stack(diagnostic, axis=0)
    labels = np.asarray([r["labels"] for r in val_rows], dtype=np.uint8)
    # Check the single frozen split uses the same label rows already bound by v1 models.
    for seed in SEEDS:
        y = np.load(N / f"seed_{seed}/validation.npz", allow_pickle=False)["y"]
        if y.shape != labels.shape or not np.array_equal(y, labels):
            raise RuntimeError(f"validation labels/order differ from frozen ResNet seed {seed}")
    target = OUT / "huang_validation_scores.npz"
    tmp = target.with_name("." + target.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, record_ids=val_ids, labels=labels, detector_probabilities=validation_detector["probabilities"],
                        predicted_class=val_classes, diagnostic_scores=score, seeds=np.asarray(SEEDS),
                        endpoints=np.asarray(["TWA", "SB", "STach"]))
    os.replace(tmp, target)
    report = {
        "status": "PASS",
        "protocol_sha256": sha(protocol_path),
        "amendment_sha256": sha(amendment_path) if amendment is not None else None,
        "detector_replay": {"n": 64, "max_abs_probability_delta": detector_delta, "tolerance": detector_tol},
        "diagnostic_replay": diag_replay,
        "validation_records": len(val_rows),
        "validation_record_order_sha256": hashlib.sha256("\n".join(val_ids.tolist()).encode()).hexdigest(),
        "validation_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "validation_detector_probabilities_sha256": hashlib.sha256(validation_detector["probabilities"].tobytes()).hexdigest(),
        "validation_diagnostic_scores_sha256": hashlib.sha256(score.tobytes()).hexdigest(),
        "clean_validation_incorrect_correction_count": int(np.count_nonzero(val_classes != 0)),
        "clean_validation_incorrect_correction_fraction": clean_incorrect,
        "validation_detector_argmax_class_counts": {str(c): int(np.count_nonzero(val_classes == c)) for c in range(4)},
        "validation_score_path": target.relative_to(ROOT).as_posix(),
        "validation_score_file_sha256": sha(target),
        "cuda_device": torch.cuda.get_device_name(0),
        "seconds": round(time.monotonic() - start, 3),
        "test_labels_not_used_for_fitting": True,
        "training": False,
    }
    (OUT / "HUANG_VALIDATION_INFERENCE_AUDIT.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
