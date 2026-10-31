"""Read and validate the restricted HEEDB WFDB sample without model inference."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


ROOT = Path(__file__).resolve().parents[1]
RESTRICTED = ROOT / "data" / "heedb_restricted" / "I0006"
CANDIDATES = RESTRICTED / "audit_manifests" / "waveform_sample_candidates.csv"
OBJECTS = RESTRICTED / "audit_manifests" / "waveform_object_manifest.csv"
DETAILS = RESTRICTED / "audit_manifests" / "waveform_smoke_results.csv"
OUTPUT = ROOT / "outputs" / "heedb_readiness_v1" / "WAVEFORM_SMOKE_SUMMARY.json"

CORE = {
    "TWA": [1140, 1145, 1150, 1151, 1160, 1161, 1170, 1171, 1172, 1180, 1181],
    "SB": [21, 24],
    "STach": [23],
}


def contains_code(raw: str, codes: list[int]) -> int:
    observed = {int(token.strip()) for token in raw.split(",") if token.strip().isdigit()}
    return int(bool(observed & set(codes)))


def main() -> None:
    sample = pd.read_csv(CANDIDATES, dtype=str, keep_default_na=False)
    objects = pd.read_csv(OBJECTS, dtype=str, keep_default_na=False)
    metadata = pd.read_csv(
        RESTRICTED / "metadata" / "metadata.csv",
        usecols=["BDSPPatientID", "FileName", "FileID"],
        dtype=str,
        keep_default_na=False,
    )
    diagnoses = pd.read_csv(
        RESTRICTED / "12SL_diagnoses" / "diagnoses_acquisition.csv",
        usecols=["FileName", "codes_physician", "codes_software"],
        dtype=str,
        keep_default_na=False,
    )
    joined = sample.merge(
        metadata,
        on=["BDSPPatientID", "FileName", "FileID"],
        how="left",
        validate="one_to_one",
        indicator="metadata_join",
        suffixes=("", "_metadata"),
    ).merge(
        diagnoses,
        on=["FileName", "codes_physician", "codes_software"],
        how="left",
        validate="one_to_one",
        indicator="diagnosis_join",
        suffixes=("", "_diagnosis"),
    )

    results: list[dict] = []
    for row in joined.itertuples(index=False):
        stem = RESTRICTED / "waveform_sample" / row.FileName
        record = wfdb.rdrecord(str(stem), physical=True)
        signal = np.asarray(record.p_signal)
        expected_bits = "".join(str(contains_code(row.codes_physician, CORE[name])) for name in CORE)
        results.append(
            {
                "BDSPPatientID": row.BDSPPatientID,
                "FileName": row.FileName,
                "metadata_join": row.metadata_join,
                "diagnosis_join": row.diagnosis_join,
                "stored_label_pattern": row.label_pattern_TWA_SB_STach,
                "recomputed_label_pattern": expected_bits,
                "fs_hz": float(record.fs),
                "samples": int(record.sig_len),
                "signals": int(record.n_sig),
                "lead_order": "|".join(record.sig_name),
                "units": "|".join(record.units or []),
                "nan_samples": int(np.isnan(signal).sum()),
                "infinite_samples": int(np.isinf(signal).sum()),
                "minimum_finite": float(np.nanmin(signal)),
                "maximum_finite": float(np.nanmax(signal)),
            }
        )

    detail = pd.DataFrame(results)
    DETAILS.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAILS, index=False)

    expected_order = "I|II|III|aVR|aVL|aVF|V1|V2|V3|V4|V5|V6"
    waveform_bytes = int(pd.to_numeric(objects["remote_bytes"]).sum())
    summary = {
        "records": int(len(detail)),
        "patients": int(detail["BDSPPatientID"].nunique()),
        "metadata_joined": int(detail["metadata_join"].eq("both").sum()),
        "diagnoses_joined": int(detail["diagnosis_join"].eq("both").sum()),
        "label_recomputed_equal": int(
            detail["stored_label_pattern"].eq(detail["recomputed_label_pattern"]).sum()
        ),
        "readable_records": int(len(detail)),
        "fs_hz_counts": {str(k): int(v) for k, v in detail["fs_hz"].value_counts().items()},
        "sample_count_distribution": {
            str(k): int(v) for k, v in detail["samples"].value_counts().items()
        },
        "signal_count_distribution": {
            str(k): int(v) for k, v in detail["signals"].value_counts().items()
        },
        "lead_order_counts": {
            str(k): int(v) for k, v in detail["lead_order"].value_counts().items()
        },
        "expected_lead_order_records": int(detail["lead_order"].eq(expected_order).sum()),
        "unit_order_counts": {str(k): int(v) for k, v in detail["units"].value_counts().items()},
        "records_with_nan_samples": int(detail["nan_samples"].gt(0).sum()),
        "total_nan_samples": int(detail["nan_samples"].sum()),
        "records_with_infinite_samples": int(detail["infinite_samples"].gt(0).sum()),
        "total_infinite_samples": int(detail["infinite_samples"].sum()),
        "label_pattern_counts": {
            str(k): int(v) for k, v in detail["stored_label_pattern"].value_counts().sort_index().items()
        },
        "waveform_object_count": int(len(objects)),
        "waveform_bytes": waveform_bytes,
        "all_object_sizes_locally_verified": bool(objects["local_verified"].eq("1").all()),
        "restricted_detail_manifest": str(DETAILS.relative_to(ROOT)),
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
