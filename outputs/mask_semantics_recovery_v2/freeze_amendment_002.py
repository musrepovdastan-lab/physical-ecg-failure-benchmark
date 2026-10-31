"""Bind the bootstrap cache-path shadowing correction before resuming analysis."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.mask_semantics_recovery_v2 as recovery

out = ROOT / "outputs/mask_semantics_recovery_v2"
protocol = json.loads((out / "RECOVERY_PROTOCOL.json").read_text(encoding="utf-8"))
amendment1 = json.loads((out / "RECOVERY_PROTOCOL_AMENDMENT_001.json").read_text(encoding="utf-8"))
obj = {
    "amendment_version": "002",
    "amended_utc": datetime.now(timezone.utc).isoformat(),
    "prior_protocol_binding_sha256": protocol["binding_sha256"],
    "prior_amendment_binding_sha256": amendment1["amendment_binding_sha256"],
    "prior_runner_source_sha256": amendment1["runner_source_sha256"],
    "runner_source_sha256": recovery.sha256(Path(recovery.__file__)),
    "reason": "The point-metric stage completed, but bootstrap task validation exposed variable shadowing: the last prediction cache path was checked in place of the bootstrap task path. No bootstrap task was accepted or written before this validation failure.",
    "change": "Keep prediction-cache hashes in prediction_path and validate/write the separate bootstrap task_path. Preserve prediction checks, bootstrap indices, mask semantics, frozen thresholds, and all generated point metrics.",
    "scope": "Bootstrap cache-path bookkeeping only; no scores, estimands, labels, split, seed, model, training, or tolerance changed.",
    "existing_mask_inference_preserved": True,
    "point_metrics_completed_before_patch": True,
}
obj["amendment_binding_sha256"] = recovery.hash_json(obj)
(out / "RECOVERY_PROTOCOL_AMENDMENT_002.json").write_text(
    json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(obj["amendment_binding_sha256"], obj["runner_source_sha256"])
