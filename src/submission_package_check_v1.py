"""Read-only, independent consistency check of aggregate submission figures."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "submission_evidence_v1"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: str, expected: str) -> bool:
    return abs(float(actual) - float(expected)) < 1e-12


def main() -> None:
    manifest = json.loads((OUT / "FIGURE_MANIFEST.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["source_hashes"].items():
        assert sha(ROOT / relative) == expected, relative
    for relative, expected in manifest["artifact_hashes"].items():
        assert sha(OUT / relative) == expected, relative
    data = rows(OUT / "FIGURE_DATA.csv")
    assert len(data) == manifest["figure_data_rows"] == 64
    assert set(data[0]) == {"figure", "dataset", "arm", "seed", "condition", "metric", "value", "low95", "high95"}
    assert {r["figure"] for r in data} == {"F1", "F2", "F3"}
    keys = [(r["figure"], r["dataset"], r["arm"], r["seed"], r["condition"], r["metric"]) for r in data]
    assert len(set(keys)) == len(keys)

    prior = rows(ROOT / "outputs/post_m2_paper_readiness_v2/CROSS_DATASET_PRIMARY_RESULTS.csv")
    lookup = {(r["dataset"], r["arm"], r["seed"]): r for r in prior}
    f1 = [r for r in data if r["figure"] == "F1"]
    assert len(f1) == 36
    columns = {"clean": "clean_twa_auprc", "mean_792_five_lead_masks": "mean_five_lead_mask_twa_auprc",
               "RA_LL_reversal": "ra_ll_twa_auprc"}
    for row in f1:
        source = lookup[row["dataset"], row["arm"], row["seed"]]
        assert row["metric"] == "TWA_AUPRC" and close(row["value"], source[columns[row["condition"]]])

    heedb = rows(ROOT / "outputs/heedb_huang_exploratory_v1/TWA_THREE_SEED_MEANS.csv")
    lookup_h = {(r["arm"], r["condition"]): r for r in heedb}
    f2 = [r for r in data if r["figure"] == "F2"]
    assert len(f2) == 24
    for row in f2:
        assert row["dataset"] == "HEEDB_independent" and row["seed"] == "three_seed_mean"
        assert close(row["value"], lookup_h[row["arm"], row["condition"]][row["metric"]])

    ci = rows(ROOT / "outputs/heedb_huang_exploratory_v1/PAIRED_PATIENT_BOOTSTRAP.csv")
    f3 = [r for r in data if r["figure"] == "F3"]
    assert len(f3) == 4
    for row in f3:
        matched = [r for r in ci if r["type"] == "paired_contrast" and r["left"] == "physics_aware"
                   and r["right"] == "huang_detect_correct_clean_resnet"
                   and r["condition"] == "RA_LL_reversal" and r["metric"] == row["metric"]]
        assert len(matched) == 1
        assert all(close(row[a], matched[0][b]) for a, b in (("value", "estimate"), ("low95", "low95"), ("high95", "high95")))
    print(json.dumps({"status": "PASS", "figure_rows": len(data), "source_hashes": len(manifest["source_hashes"]),
                      "artifact_hashes": len(manifest["artifact_hashes"]), "figures": len(list((OUT / "figures").glob("*.svg")))},
                     sort_keys=True))


if __name__ == "__main__":
    main()
