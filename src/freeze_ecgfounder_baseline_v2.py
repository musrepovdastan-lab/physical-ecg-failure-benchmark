"""Freeze v2 after complete input audit and training-only smoke, before outcomes."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/ecgfounder_baseline_v2"
DEST = ROOT / "configs/FROZEN_ECGFOUNDER_BASELINE_PROTOCOL_v2.json"
V1 = ROOT / "configs/FROZEN_ECGFOUNDER_BASELINE_PROTOCOL_v1.json"
V1_SHA = "17214328ef9e0443692da4248df7683cff906689d270b3e08efe3311710c1bee"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if DEST.exists():
        raise RuntimeError("v2 protocol already frozen")
    if sha(V1) != V1_SHA:
        raise RuntimeError("v1 protocol changed")
    audit_path = OUT / "PRE_RUN_AUDIT.json"
    missing_path = OUT / "MISSING_PATH_AUDIT.json"
    smoke_path = OUT / "PREFREEZE_SMOKE.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["total_records"] != 45133 or smoke["status"] != "PASS":
        raise RuntimeError("v2 preflight or smoke gate failed")
    known = {r["record_id"] for r in audit["known_failure_cases"]}
    if known != {"g1/JS10767", "g11/JS21646", "g13/JS23074"}:
        raise RuntimeError("known failures were not verified")
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    for rel, digest in v1["file_bindings"].items():
        if sha(ROOT / rel) != digest:
            raise RuntimeError(f"v1 binding changed: {rel}")
    cfg = json.loads(V1.read_text(encoding="utf-8"))
    cfg["version"] = "ecgfounder_baseline_v2"
    cfg["status"] = "FROZEN_AFTER_INPUT_ADAPTER_PREFLIGHT_BEFORE_NEW_OUTCOMES"
    cfg["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    cfg["parent_protocol_sha256"] = V1_SHA
    cfg["amendment_scope"] = "input adapter only; all scientific choices copied byte-for-byte as JSON values from v1"
    cfg["freeze_access_status"] = {
        "test_inputs_audited_before_freeze": True,
        "test_input_audit_scope": "paths, headers, raw waveform hashes and missing-value counts only",
        "test_model_outcomes_evaluated_before_freeze": False,
        "test_labels_or_predictions_used_to_select_v2_head_before_freeze": False,
    }
    cfg["input_adapter_v2"] = {
        "ningbo_path": "resolve every ID through canonical header_path and waveform_path; header filename may differ from waveform basename if its 12 signal lines reference the canonical waveform; reject missing files or ID mismatch",
        "ningbo_hash": "verify every actual waveform SHA-256 against canonical record and frozen split before full run; check ID/hash binding again during loading",
        "physical_nan": "replace WFDB NaN physical samples with zero before RA-LL permutation or any filter, as in pinned dataset.py",
        "unexpected": "reject infinities, unrecognized dimensions/rate/leads and hash mismatches; exclude no records",
        "chapman_v1_reuse": "none; recompute all Chapman features and validation-selected head in v2",
    }
    cfg["preprocessing"]["nonfinite_or_schema_mismatch"] = "WFDB NaN -> 0 before fault/filter; reject infinity and other schema errors"
    cfg["preprocessing"]["order"] = ["replace WFDB NaN physical samples with zero"] + cfg["preprocessing"]["order"]
    cfg["runtime"]["cache"] = "v2-only resumable feature and logit arrays bound to v2 protocol/checkpoint/ordered IDs; completed hashes required before reuse"
    cfg["runner_sha256"] = sha(ROOT / "src/ecgfounder_baseline_v2.py")
    cfg["analysis_sha256"] = sha(ROOT / "src/ecgfounder_analyze_v2.py")
    cfg["tests_sha256"] = sha(ROOT / "tests/test_ecgfounder_baseline_v2.py")
    cfg["pre_run_audit_sha256"] = sha(audit_path)
    cfg["prefreeze_smoke_sha256"] = sha(smoke_path)
    cfg["missing_path_audit_sha256"] = sha(missing_path)
    cfg["file_bindings"].update({
        "src/ecgfounder_preflight_v2.py": sha(ROOT / "src/ecgfounder_preflight_v2.py"),
        "src/ecgfounder_prefreeze_smoke_v2.py": sha(ROOT / "src/ecgfounder_prefreeze_smoke_v2.py"),
        "src/ecgfounder_analyze_v2.py": cfg["analysis_sha256"],
        "outputs/ecgfounder_baseline_v2/MISSING_PATH_AUDIT.json": sha(missing_path),
        "outputs/ecgfounder_baseline_v2/PREFREEZE_SMOKE.json": sha(smoke_path),
    })
    DEST.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "FROZEN_PROTOCOL_HASH.json").write_text(json.dumps({
        "protocol_sha256": sha(DEST), "parent_protocol_sha256": V1_SHA,
        "frozen_utc": cfg["frozen_utc"],
        "test_inputs_audited_before_freeze": True,
        "test_model_outcomes_evaluated_before_freeze": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(sha(DEST))


if __name__ == "__main__":
    main()
