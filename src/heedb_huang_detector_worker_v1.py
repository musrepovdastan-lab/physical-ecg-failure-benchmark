"""Isolated TensorFlow worker for the exploratory, frozen HEEDB Huang detector."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.heedb_huang_adapter_v1 import detector_input, electrical_reversal
from src.huang_detector_infer_v2 import load_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "heedb_huang_exploratory_v1"
PRIVATE = ROOT / "data" / "heedb_restricted" / "I0006" / "huang_exploratory_v1"
SOURCE = ROOT / "data" / "heedb_restricted" / "I0006" / "independent_confirmation_v1" / "signals_10000_float32.npy"
PROTOCOL = OUT / "FROZEN_PROTOCOL.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def log(stage: str, **fields: object) -> None:
    row = {"utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **fields}
    with (OUT / "PROGRESS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if digest(PROTOCOL) != (OUT / "FROZEN_PROTOCOL_SHA256.txt").read_text(encoding="utf-8").strip():
        raise RuntimeError("frozen comparator protocol changed")
    if digest(Path(__file__)) != protocol["worker_sha256"] or digest(SOURCE) != protocol["cache_sha256"]:
        raise RuntimeError("worker or source waveform cache changed")
    if digest(ROOT / "src" / "heedb_huang_adapter_v1.py") != protocol["adapter_sha256"]:
        raise RuntimeError("HEEDB-to-Huang adapter changed")
    for rel, expected in protocol["detector_weights_sha256"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError("published detector weights changed")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    final = PRIVATE / "detector_probabilities_float32.npy"
    partial = PRIVATE / "detector_probabilities_float32.npy.partial"
    if final.exists() or partial.exists():
        raise RuntimeError("detector output already exists; refusing a duplicate run")
    cache = np.load(SOURCE, mmap_mode="r")
    if cache.shape != (10000, 12, 5000) or cache.dtype != np.float32:
        raise RuntimeError("source waveform geometry changed")
    model, checkpoint, tf_version = load_model()
    result = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(2, 10000, 4))
    started = time.monotonic()
    for condition_index, condition in enumerate(("clean", "RA_LL_reversal")):
        for start in range(0, 10000, 32):
            stop = min(start + 32, 10000)
            batch = []
            for record_index in range(start, stop):
                raw = np.asarray(cache[record_index])
                if condition_index:
                    raw = electrical_reversal(raw, "RA_LL_reversal")
                batch.append(detector_input(raw))
            p = np.asarray(model.predict(np.asarray(batch, dtype=np.float32), verbose=0), dtype=np.float32)
            if p.shape != (stop - start, 4) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-4):
                raise RuntimeError("published detector returned invalid four-class probabilities")
            result[condition_index, start:stop] = p
            if stop % 320 == 0 or stop == 10000:
                result.flush()
                log("DETECTOR_PROGRESS", condition=condition, completed=stop, total=10000)
    result.flush()
    del result
    os.replace(partial, final)
    (OUT / "DETECTOR_COMPLETE.json").write_text(json.dumps({
        "decision": "PASS", "patients": 10000, "conditions": 2,
        "detector_probabilities_sha256": digest(final), "tensorflow_version": tf_version,
        "checkpoint_prefix": str(Path(checkpoint).relative_to(ROOT)),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log("DETECTOR_COMPLETE", patients=10000, elapsed_seconds=round(time.monotonic() - started, 3))


if __name__ == "__main__":
    main()
