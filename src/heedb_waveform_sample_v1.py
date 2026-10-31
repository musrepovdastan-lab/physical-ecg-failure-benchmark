"""Create a deterministic, patient-disjoint HEEDB readability sample manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heedb_restricted" / "I0006"
OUTPUT = DATA / "audit_manifests" / "waveform_sample_candidates.csv"
SEED = "heedb_readiness_v1|2026-09-24|patient-disjoint|three-per-stratum"

CORE = {
    "TWA": [1140, 1145, 1150, 1151, 1160, 1161, 1170, 1171, 1172, 1180, 1181],
    "SB": [21, 24],
    "STach": [23],
}


def has_code(series: pd.Series, codes: list[int]) -> pd.Series:
    expression = r"(?<!\d)(?:" + "|".join(str(code) for code in codes) + r")(?!\d)"
    return series.str.contains(expression, regex=True, na=False)


def rank_key(stratum: str, patient_id: str, file_name: str) -> str:
    value = f"{SEED}|{stratum}|{patient_id}|{file_name}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    metadata = pd.read_csv(
        DATA / "metadata" / "metadata.csv",
        usecols=["BDSPPatientID", "FileName", "FileID"],
        dtype=str,
        keep_default_na=False,
    )
    diagnoses = pd.read_csv(
        DATA / "12SL_diagnoses" / "diagnoses_acquisition.csv",
        usecols=["FileName", "codes_software", "codes_physician"],
        dtype=str,
        keep_default_na=False,
    )
    frame = metadata.merge(diagnoses, on="FileName", how="inner", validate="one_to_one")
    frame = frame.loc[frame["BDSPPatientID"].ne("")].copy()
    for outcome, codes in CORE.items():
        frame[outcome] = has_code(frame["codes_physician"], codes).astype(int)
    frame["label_pattern_TWA_SB_STach"] = (
        frame[["TWA", "SB", "STach"]].astype(str).agg("".join, axis=1)
    )

    chosen = []
    used_patients: set[str] = set()
    for stratum in ["000", "001", "010", "011", "100", "101", "110", "111"]:
        candidates = frame.loc[frame["label_pattern_TWA_SB_STach"].eq(stratum)].copy()
        target = min(3, int(candidates["BDSPPatientID"].nunique()))
        if target == 0:
            raise RuntimeError(f"No patient-linked candidate exists for recording stratum {stratum}")
        candidates["selection_sha256"] = [
            rank_key(stratum, patient_id, file_name)
            for patient_id, file_name in zip(candidates["BDSPPatientID"], candidates["FileName"])
        ]
        candidates = candidates.sort_values(["selection_sha256", "FileName"], kind="mergesort")
        for row in candidates.itertuples(index=False):
            if row.BDSPPatientID in used_patients:
                continue
            chosen.append(row._asdict())
            used_patients.add(row.BDSPPatientID)
            if sum(item["label_pattern_TWA_SB_STach"] == stratum for item in chosen) == target:
                break

    sample = pd.DataFrame(chosen)
    if (
        len(sample) > 24
        or sample["BDSPPatientID"].nunique() != len(sample)
        or sample["FileName"].nunique() != len(sample)
        or sample["label_pattern_TWA_SB_STach"].nunique() != 8
    ):
        raise RuntimeError("Patient-disjoint eight-stratum sample construction failed")
    sample["remote_stem"] = "ECG/I0006/" + sample["FileName"]
    keep = [
        "BDSPPatientID",
        "FileName",
        "FileID",
        "codes_physician",
        "codes_software",
        "TWA",
        "SB",
        "STach",
        "label_pattern_TWA_SB_STach",
        "selection_sha256",
        "remote_stem",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sample[keep].to_csv(OUTPUT, index=False)
    print(f"manifest={OUTPUT}")
    print(f"patients={sample['BDSPPatientID'].nunique()} recordings={sample['FileName'].nunique()}")
    print(sample["label_pattern_TWA_SB_STach"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
