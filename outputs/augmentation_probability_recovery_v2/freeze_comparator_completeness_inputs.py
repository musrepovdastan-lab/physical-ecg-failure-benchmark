from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from outputs.augmentation_probability_study_v1 import paired_bootstrap as paths

SEEDS = (104729, 130363, 155921)
ARMS = ("ordinary", "physics_aware", "combined")
MASKS = tuple(__import__("itertools").combinations(range(12), 5))
CONDITIONS = ("clean", "RA_LL_reversal", "cycle_slots_120", "cycle_slots_201", *("mask5_" + "_".join(map(str, q)) for q in MASKS))


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    protocol = json.loads((OUT / "RECOVERY_PROTOCOL.json").read_text(encoding="utf-8"))
    files = {}
    for cohort in ("Chapman", "Ningbo"):
        for arm in ARMS:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    p = paths.path_for(cohort, arm, seed, condition)
                    if not p.is_file():
                        raise FileNotFoundError(p)
                    files[p.relative_to(ROOT).as_posix()] = sha_file(p)
    helper_files = [
        OUT / "complete_historical_comparator_fields.py",
        ROOT / "outputs/augmentation_probability_study_v1/METRICS_ALL_ARMS_BY_SEED.csv",
        ROOT / "outputs/augmentation_probability_study_v1/paired_bootstrap.py",
    ]
    helpers = {p.relative_to(ROOT).as_posix(): sha_file(p) for p in helper_files}
    supplement = {
        "package": "augmentation_probability_recovery_v2",
        "supplement": "historical comparator completeness calculation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_SUPPLEMENTARY_COMPARATOR_RECALCULATION",
        "parent_recovery_protocol_sha256": protocol["recovery_protocol_sha256"],
        "temporal_note": "This supplement was fixed after primary validation restoration and new-arm threshold/calibration recovery, but before rerunning the additional historical-comparator completeness pass. It does not claim pre-registration before original test review.",
        "scope": {"cohorts": ["Chapman", "Ningbo"], "arms": list(ARMS), "seeds": list(SEEDS), "conditions_per_unit": len(CONDITIONS), "cached_test_prediction_files": len(files), "inference_performed": False, "purpose": "recover missing derived comparator table fields from existing saved test probabilities"},
        "inputs_sha256": dict(sorted(files.items())),
        "helper_and_table_sha256": helpers,
    }
    encoded = json.dumps(supplement, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    supplement["supplement_sha256"] = sha_bytes(encoded)
    target = OUT / "COMPARATOR_COMPLETENESS_SUPPLEMENT.json"
    target.write_text(json.dumps(supplement, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"files_hashed": len(files), "supplement": str(target), "supplement_sha256": supplement["supplement_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
