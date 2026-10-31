"""Isolated published Huang limb-lead detector inference for the multifault audit."""
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
V1 = ROOT / "outputs" / "published_detector_pipeline_v1"
SOURCE = V1 / "official_source" / "ECG_lead_check-main"
LOADABLE = V1 / "official_source" / "LDenseNet_loadable"
N = ROOT / "outputs" / "ningbo_confirmation_v2"
DATA = ROOT / "data" / "challenge2021_v103"
CLASSES = ("clean", "RA_LA", "RA_LL", "LA_LL")
CONDITIONS = ("clean", "RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal")
PERMUTATIONS = {
    "RA_LA_reversal": (1, 0, 2),
    "RA_LL_reversal": (2, 1, 0),
    "LA_LL_reversal": (0, 2, 1),
}


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def canonical_path(rel: str) -> Path:
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


def records(limit: int | None) -> list[dict]:
    canonical = json.loads((N / "CANONICAL_RECORDS.json").read_text(encoding="utf8"))["records"]
    membership = json.loads((N / "SPLIT_MEMBERSHIP.json").read_text(encoding="utf8"))["rows"]
    by_id = {row["record_id"]: row for row in canonical}
    rows = [by_id[row["record_id"]] for row in membership if row["split"] == "test"]
    if len(rows) != 6981:
        raise RuntimeError("unexpected Ningbo test count")
    return rows[:limit] if limit else rows


def load_full_display(record: dict) -> np.ndarray:
    x = np.asarray(loadmat(canonical_path(record["waveform_path"]))["val"])
    if x.shape != (12, 5000):
        raise ValueError(f"expected 12x5000 display ECG, got {x.shape} for {record['record_id']}")
    return x.astype("float32", copy=False)


def display_to_independent(x: np.ndarray) -> np.ndarray:
    return np.concatenate((x[:2], x[6:]), axis=0)


def derive_display(independent: np.ndarray) -> np.ndarray:
    lead_i, lead_ii = independent[0], independent[1]
    limb = np.stack(
        (lead_i, lead_ii, lead_ii - lead_i, -(lead_i + lead_ii) / 2, lead_i - lead_ii / 2, lead_ii - lead_i / 2),
        axis=0,
    )
    return np.concatenate((limb, independent[2:]), axis=0).astype("float32")


def terminal_permutation(independent: np.ndarray, permutation: tuple[int, int, int]) -> np.ndarray:
    ra = np.zeros_like(independent[0])
    la, ll = independent[0], independent[1]
    physical = (ra, la, ll)
    slot_ra, slot_la, slot_ll = (physical[j] for j in permutation)
    return np.concatenate((np.stack((slot_la - slot_ra, slot_ll - slot_ra)), independent[2:]), axis=0).astype("float32")


def apply_condition(x: np.ndarray, condition: str) -> np.ndarray:
    if condition == "clean":
        return x
    return derive_display(terminal_permutation(display_to_independent(x), PERMUTATIONS[condition]))


def preprocess_one(x: np.ndarray) -> np.ndarray:
    # Exact implementation path used by NeuroKit signal_filter(method="butterworth", order=5).
    sos = scipy.signal.butter(5, [0.5, 35], btype="bandpass", output="sos", fs=500)
    leads = []
    for lead in x:
        y = scipy.signal.sosfiltfilt(sos, lead)
        y = scipy.signal.resample(y, 1200)
        sd = np.std(y)
        y = (y - np.mean(y)) / sd if sd != 0 else np.zeros_like(y)
        leads.append(y.astype("float32"))
    all_leads = np.asarray(leads, dtype="float32")
    return np.transpose(all_leads, (1, 0))[:, np.asarray([2, 3, -1])]


def load_model():
    sys.path.insert(0, str(SOURCE / "limb"))
    import tensorflow as tf  # noqa: PLC0415
    from LDenseNet import build_model  # noqa: PLC0415

    model = build_model((1200, 3), 4)
    checkpoint = tf.train.latest_checkpoint(str(LOADABLE))
    if checkpoint is None:
        raise FileNotFoundError(f"no checkpoint in {LOADABLE}")
    status = model.load_weights(checkpoint)
    status.assert_existing_objects_matched()
    status.expect_partial()  # Published optimizer slots are not part of the inference model.
    return model, checkpoint, tf.__version__


def predict(model, rows: list[dict], condition: str, batch_size: int) -> np.ndarray:
    outputs = []
    batch = []
    for index, record in enumerate(rows, 1):
        batch.append(preprocess_one(apply_condition(load_full_display(record), condition)))
        if len(batch) == batch_size or index == len(rows):
            outputs.append(model.predict(np.asarray(batch, dtype="float32"), verbose=0).astype("float32"))
            batch = []
            print(json.dumps({"condition": condition, "completed": index, "total": len(rows)}), flush=True)
    return np.concatenate(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    model, checkpoint, tf_version = load_model()
    if args.audit_only:
        print(json.dumps({
            "model_variable_match_assertion": "PASS",
            "checkpoint": checkpoint,
            "tensorflow_version": tf_version,
            "model_variables": len(model.variables),
            "trainable_variables": len(model.trainable_variables),
        }), flush=True)
        return
    if args.condition is None or args.output is None:
        parser.error("--condition and --output are required unless --audit-only is used")
    rows = records(args.limit)
    probabilities = predict(model, rows, args.condition, args.batch_size)
    atomic_npz(
        args.output,
        record_ids=np.asarray([row["record_id"] for row in rows]),
        class_names=np.asarray(CLASSES),
        condition=np.asarray(args.condition),
        probabilities=probabilities,
    )


if __name__ == "__main__":
    main()
