"""Train-record only v2 preprocessing and inference smoke before protocol freeze."""
import json
from pathlib import Path

import numpy as np

from src.ecgfounder_baseline_v2 import (CFG, OUT, ROOT, infer, model, physical_signal,
                                        preprocess, record_path, reverse_ra_ll)


def main():
    if CFG.exists():
        raise RuntimeError("v2 protocol already frozen")
    split = json.loads((ROOT / "outputs/ningbo_confirmation_v2/SPLIT_MEMBERSHIP.json").read_text())["rows"]
    byid = {r["record_id"]: r for r in split}
    ids = ("g1/JS10767", "g11/JS21646", "g13/JS23074")
    if any(byid[i]["split"] != "train" for i in ids):
        raise RuntimeError("known smoke IDs are not in Ningbo training split")
    net = model()
    records = []
    for record_id in ids:
        x = physical_signal("Ningbo", record_id)
        altered = reverse_ra_ll(x)
        if not np.allclose(x[6:], altered[6:], rtol=0, atol=1e-12):
            raise RuntimeError("RA-LL altered a chest lead")
        clean_input = preprocess(x)
        fault_input = preprocess(altered)
        f, logits = infer(net, np.stack((clean_input, fault_input)))
        if f.shape != (2, 1024) or logits.shape != (2, 150) or not np.isfinite(f).all() or not np.isfinite(logits).all():
            raise RuntimeError("model smoke shape or finite failure")
        records.append({"record_id": record_id, "resolved_stem": str(record_path("Ningbo", record_id).relative_to(ROOT)).replace("\\", "/"),
                        "input_shape": list(clean_input.shape), "feature_shape": list(f.shape),
                        "logit_shape": list(logits.shape), "clean_and_reversal_finite": True})
    out = {"status": "PASS", "partition": "Ningbo train only", "records": records}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PREFREEZE_SMOKE.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
