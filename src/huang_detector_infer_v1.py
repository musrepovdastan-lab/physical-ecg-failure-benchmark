"""Isolated Huang et al. 2024 limb-lead detector inference.

Run this with the TensorFlow detector environment, not the project PyTorch env.
It intentionally avoids importing neurokit2 because its package import pulls in
scikit-learn modules that fail to load on this Windows host. The filtering code
matches neurokit2.signal_filter(..., method="butterworth", order=5): scipy SOS
Butterworth plus sosfiltfilt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import scipy.signal
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "published_detector_pipeline_v1"
SOURCE = OUT / "official_source" / "ECG_lead_check-main"
LOADABLE = OUT / "official_source" / "LDenseNet_loadable"
N = ROOT / "outputs" / "ningbo_confirmation_v2"
DATA = ROOT / "data" / "challenge2021_v103"


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def path(rel: str) -> Path:
    raw = str(rel)
    if "\\" in raw:
        raise ValueError("noncanonical path separator")
    p = PurePosixPath(raw)
    if p.is_absolute() or ":" in raw or ".." in p.parts or p.parts[:2] != ("training", "ningbo"):
        raise ValueError("invalid Ningbo mapping path")
    q = (DATA / "ningbo").joinpath(*p.parts[2:])
    if not q.is_file():
        raise FileNotFoundError(str(q))
    return q


def load_full_display(record: dict) -> np.ndarray:
    x = np.asarray(loadmat(path(record["waveform_path"]))["val"])
    if x.shape != (12, 5000):
        raise ValueError(f"expected 12x5000 display ECG, got {x.shape} for {record['record_id']}")
    return x.astype("float32", copy=False)


def display_to_independent(x: np.ndarray) -> np.ndarray:
    return np.concatenate((x[:2], x[6:]), axis=0)


def derive_display(independent: np.ndarray) -> np.ndarray:
    i, ii = independent[0], independent[1]
    limb = np.stack((i, ii, ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2), axis=0)
    return np.concatenate((limb, independent[2:]), axis=0).astype("float32")


def terminal_permutation(independent: np.ndarray, permutation: tuple[int, int, int]) -> np.ndarray:
    ra = np.zeros_like(independent[0])
    la, ll = independent[0], independent[1]
    physical = (ra, la, ll)
    slot_ra, slot_la, slot_ll = (physical[j] for j in permutation)
    new_i = slot_la - slot_ra
    new_ii = slot_ll - slot_ra
    return np.concatenate((np.stack((new_i, new_ii), axis=0), independent[2:]), axis=0).astype("float32")


def apply_fault(x: np.ndarray, condition: str) -> np.ndarray:
    if condition == "clean":
        return x
    if condition != "RA_LL_reversal":
        raise ValueError(condition)
    return derive_display(terminal_permutation(display_to_independent(x), (2, 1, 0)))


def preprocess_one(x: np.ndarray) -> np.ndarray:
    leads = []
    sos = scipy.signal.butter(5, [0.5, 35], btype="bandpass", output="sos", fs=500)
    for lead in x:
        y = scipy.signal.sosfiltfilt(sos, lead)
        y = scipy.signal.resample(y, 1200)
        sd = np.std(y)
        y = (y - np.mean(y)) / sd if sd != 0 else np.zeros_like(y)
        leads.append(y.astype("float32"))
    z = np.asarray(leads, dtype="float32")
    return np.transpose(z, (1, 0))[:, np.asarray([2, 3, -1])]


def records(limit: int | None) -> list[dict]:
    can = json.loads((N / "CANONICAL_RECORDS.json").read_text(encoding="utf8"))["records"]
    split = json.loads((N / "SPLIT_MEMBERSHIP.json").read_text(encoding="utf8"))["rows"]
    by = {r["record_id"]: r for r in can}
    out = [by[r["record_id"]] for r in split if r["split"] == "test"]
    if len(out) != 6981:
        raise RuntimeError("unexpected Ningbo test count")
    return out[:limit] if limit else out


def load_model():
    sys.path.insert(0, str(SOURCE / "limb"))
    import tensorflow as tf  # noqa: PLC0415
    from LDenseNet import build_model  # noqa: PLC0415

    model = build_model((1200, 3), 4)
    model.load_weights(tf.train.latest_checkpoint(str(LOADABLE))).expect_partial()
    return model


def predict_condition(model, rows: list[dict], condition: str, batch_size: int) -> np.ndarray:
    probs = []
    batch = []
    for i, record in enumerate(rows, 1):
        x = apply_fault(load_full_display(record), condition)
        batch.append(preprocess_one(x))
        if len(batch) == batch_size or i == len(rows):
            p = model.predict(np.asarray(batch, dtype="float32"), verbose=0)
            probs.append(p.astype("float32"))
            batch = []
            print(json.dumps({"condition": condition, "completed": i, "total": len(rows)}), flush=True)
    return np.concatenate(probs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = records(args.limit)
    model = load_model()
    clean = predict_condition(model, rows, "clean", args.batch_size)
    ra_ll = predict_condition(model, rows, "RA_LL_reversal", args.batch_size)
    atomic_npz(
        args.output,
        record_ids=np.asarray([r["record_id"] for r in rows]),
        class_names=np.asarray(["clean", "RA_LA", "RA_LL", "LA_LL"]),
        probs_clean=clean,
        probs_ra_ll=ra_ll,
    )


if __name__ == "__main__":
    main()
