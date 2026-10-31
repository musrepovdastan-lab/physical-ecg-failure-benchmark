"""Read-only audit of WFDB's digital missing-value sentinel on training data."""
import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

root = Path(__file__).resolve().parents[2]
rows = [r for r in json.loads((root / "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json").read_text())["rows"]
        if r["split"] == "train"]
base = root / "data/challenge2021_v103/ningbo"
canonical = {r["record_id"]: r for r in json.loads((root / "outputs/ningbo_confirmation_v2/CANONICAL_RECORDS.json").read_text())["records"]}
count = 0
samples = 0
examples = []
started = time.time()
for r in rows:
    relative = canonical[r["record_id"]]["waveform_path"]
    assert relative.startswith("training/ningbo/")
    x = loadmat(base / relative.removeprefix("training/ningbo/"))["val"]
    n = int(np.count_nonzero(x == -32768))
    if n:
        count += 1
        samples += n
        if len(examples) < 10:
            examples.append({"record_id": r["record_id"], "sentinel_samples": n})
result = {"split": "training_only", "records_scanned": len(rows),
          "records_with_minus32768": count, "total_minus32768_samples": samples,
          "first_examples": examples, "elapsed_seconds": round(time.time() - started, 2)}
dest = root / "outputs/ecgfounder_baseline_v1/NINGBO_TRAIN_MISSING_AUDIT.json"
dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
