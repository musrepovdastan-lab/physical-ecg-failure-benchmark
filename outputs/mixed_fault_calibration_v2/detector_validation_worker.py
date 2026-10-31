"""Run the already-published detector on the frozen Ningbo validation order."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from src import huang_detector_infer_v2 as infer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
N = ROOT / "outputs" / "ningbo_confirmation_v2"
OUT = ROOT / "outputs" / "mixed_fault_calibration_v2"


def ordered_rows(split: str) -> list[dict]:
    canonical = json.loads((N / "CANONICAL_RECORDS.json").read_text(encoding="utf-8"))["records"]
    membership = json.loads((N / "SPLIT_MEMBERSHIP.json").read_text(encoding="utf-8"))["rows"]
    by_id = {row["record_id"]: row for row in canonical}
    rows = [by_id[row["record_id"]] for row in membership if row["split"] == split]
    expected = {"validation": 3488, "test": 6981}[split]
    if len(rows) != expected:
        raise RuntimeError(f"unexpected {split} record count {len(rows)} != {expected}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    rows = ordered_rows(args.split)
    if args.limit is not None:
        rows = rows[:args.limit]
    model, checkpoint, tf_version = infer.load_model()
    probabilities = infer.predict(model, rows, "clean", args.batch_size)
    ids = np.asarray([row["record_id"] for row in rows])
    tmp = args.output.with_name("." + args.output.name + f".{os.getpid()}.tmp.npz")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tmp, record_ids=ids, probabilities=probabilities, class_names=np.asarray(infer.CLASSES))
    os.replace(tmp, args.output)
    print(json.dumps({"status": "PASS", "split": args.split, "n": len(rows), "output": str(args.output),
                      "checkpoint": checkpoint, "tensorflow_version": tf_version}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
