"""Record the pilot-driven runner amendment before continuing L1 inference."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.mask_semantics_recovery_v2 as recovery

out = ROOT / "outputs/mask_semantics_recovery_v2"
protocol = json.loads((out / "RECOVERY_PROTOCOL.json").read_text(encoding="utf-8"))
obj = {
    "amendment_version": "001",
    "amended_utc": datetime.now(timezone.utc).isoformat(),
    "prior_protocol_binding_sha256": protocol["binding_sha256"],
    "prior_runner_source_sha256": protocol["evaluator"]["source_sha256"],
    "runner_source_sha256": recovery.sha256(Path(recovery.__file__)),
    "reason": "The fixed clean/RA-LL probe found 2 comparisons outside the predeclared tolerance among 1,536 scores for M2 seed 104729. Repeated recovery inference was bitwise deterministic; the maximum difference was 3.8147e-05 at a score near 0.01045. The tolerance remains unchanged.",
    "change": "Preserve the original frozen protocol. Re-infer the full clean and RA-LL score arrays for this one model/seed in the recovery directory, bound to the ordered test membership and current checkpoint/preprocessing. Reuse the other five model/seed score pairs that passed the probe. The mask transform and scientific estimands are unchanged.",
    "scope": "Runner/cache handling only; no mask semantics, thresholds, labels, split, bootstrap indices, model, or training change.",
    "initial_probe_inference_used_prior_runner": True,
    "no_corrected_mask_inference_started_before_amendment": True,
}
obj["amendment_binding_sha256"] = recovery.hash_json(obj)
path = out / "RECOVERY_PROTOCOL_AMENDMENT_001.json"
path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(obj["amendment_binding_sha256"], obj["runner_source_sha256"])
