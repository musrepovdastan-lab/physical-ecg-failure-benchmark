"""Bind the bootstrap AP matrix orientation correction before aggregation."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.mask_semantics_recovery_v2 as recovery

out = ROOT / "outputs/mask_semantics_recovery_v2"
protocol = json.loads((out / "RECOVERY_PROTOCOL.json").read_text(encoding="utf-8"))
amendment4 = json.loads((out / "RECOVERY_PROTOCOL_AMENDMENT_004.json").read_text(encoding="utf-8"))
obj = {
    "amendment_version": "005",
    "amended_utc": datetime.now(timezone.utc).isoformat(),
    "prior_protocol_binding_sha256": protocol["binding_sha256"],
    "prior_amendment_binding_sha256": amendment4["amendment_binding_sha256"],
    "prior_runner_source_sha256": amendment4["runner_source_sha256"],
    "runner_source_sha256": recovery.sha256(Path(recovery.__file__)),
    "reason": "All 18 bootstrap tasks completed, but final aggregation found the saved mask AP matrix was condition × resample while the immutable consumer contract is resample × condition. The per-mask values, exact-mask series, clean/RA-LL series, and binding hashes are valid.",
    "change": "Save and consume mask AP matrices in resample × mask order. Transpose the 18 already-computed task matrices without changing values; retain original-orientation backups and all bootstrap bindings.",
    "scope": "Bootstrap matrix orientation only; no predictions, bootstrap samples, endpoint values, label, split, contrast, frozen threshold, mask semantics, model, training, or tolerance changed.",
    "point_metrics_and_mask_inference_preserved": True,
    "bootstrap_task_binding_preserved": True,
}
obj["amendment_binding_sha256"] = recovery.hash_json(obj)
(out / "RECOVERY_PROTOCOL_AMENDMENT_005.json").write_text(
    json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(obj["amendment_binding_sha256"], obj["runner_source_sha256"])
