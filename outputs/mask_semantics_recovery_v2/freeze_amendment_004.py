"""Bind source-compatible validation for regenerated clean/RA-LL cache files."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.mask_semantics_recovery_v2 as recovery

out = ROOT / "outputs/mask_semantics_recovery_v2"
protocol = json.loads((out / "RECOVERY_PROTOCOL.json").read_text(encoding="utf-8"))
amendment3 = json.loads((out / "RECOVERY_PROTOCOL_AMENDMENT_003.json").read_text(encoding="utf-8"))
obj = {
    "amendment_version": "004",
    "amended_utc": datetime.now(timezone.utc).isoformat(),
    "prior_protocol_binding_sha256": protocol["binding_sha256"],
    "prior_amendment_binding_sha256": amendment3["amendment_binding_sha256"],
    "prior_runner_source_sha256": amendment3["runner_source_sha256"],
    "runner_source_sha256": recovery.sha256(Path(recovery.__file__)),
    "reason": "Resume validation confirmed all mask-panel caches under the frozen inference evaluator. The two clean/RA-LL arrays regenerated after the probe use the same frozen inference evaluator digest and need that digest accepted for provenance checks.",
    "change": "Apply the same exact source-digest compatibility check from Amendment 003 to verified clean and RA-LL recovery caches, while keeping protocol, model, preprocessing, ordered test membership, labels, schema, dimensions, and finite-value checks mandatory.",
    "scope": "Verified clean/RA-LL cache validation only; no predictions, metrics, mask semantics, estimands, thresholds, bootstrap indices, split, model, training, or tolerance changed.",
    "mask_prediction_compatibility_preserved": True,
    "bootstrap_task_path_fix_preserved": True,
}
obj["amendment_binding_sha256"] = recovery.hash_json(obj)
(out / "RECOVERY_PROTOCOL_AMENDMENT_004.json").write_text(
    json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(obj["amendment_binding_sha256"], obj["runner_source_sha256"])
