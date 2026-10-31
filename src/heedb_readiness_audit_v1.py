"""Aggregate-only HEEDB I0006 table audit.

Patient- and recording-level values are never written outside the restricted data
directory. The public output contains schemas, cardinalities, missingness, and
aggregate label counts only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heedb_restricted" / "I0006"
OUTPUT = ROOT / "outputs" / "heedb_readiness_v1" / "AUDIT_SUMMARY.json"

PATHS = {
    "dictionary": DATA / "12SL_diagnoses" / "diagnoses_dictionary.csv",
    "acquisition": DATA / "12SL_diagnoses" / "diagnoses_acquisition.csv",
    "v24": DATA / "12SL_diagnoses" / "diagnoses_v24.csv",
    "metadata": DATA / "metadata" / "metadata.csv",
}

# This is a deliberately explicit audit crosswalk, not a claim of SNOMED
# equivalence. "Core" adds clear named subtypes; "ambiguous" remains excluded.
CODE_GROUPS = {
    "TWA": {
        "snomed": "164934002",
        "strict": [1140],
        "core": [1140, 1145, 1150, 1151, 1160, 1161, 1170, 1171, 1172, 1180, 1181],
        "ambiguous": [1141, 1142, 1182, 1215, 1217, 1218, 1219, 1220, 1221, 1222, 1223],
        "excluded_comparison_or_fragment": [1200, 1201, 1203, 1207, 1208, 1210, 1211, 1212, 1213, 1214, 1216, 1224],
    },
    "SB": {
        "snomed": "426177001",
        "strict": [21],
        "core": [21, 24],
        "ambiguous": [265],
        "excluded_comparison_or_fragment": [],
    },
    "STach": {
        "snomed": "427084000",
        "strict": [23],
        "core": [23],
        "ambiguous": [],
        "excluded_comparison_or_fragment": [],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_or_empty": {c: int(frame[c].eq("").sum()) for c in frame.columns},
        "unique_nonempty": {
            c: int(frame.loc[frame[c].ne(""), c].nunique(dropna=False)) for c in frame.columns
        },
    }


def code_mask(series: pd.Series, codes: list[int]) -> pd.Series:
    if not codes:
        return pd.Series(False, index=series.index)
    expression = r"(?<!\d)(?:" + "|".join(str(code) for code in codes) + r")(?!\d)"
    return series.str.contains(expression, regex=True, na=False)


def token_counter(series: pd.Series) -> tuple[Counter[str], int, int]:
    counts: Counter[str] = Counter()
    malformed = 0
    duplicate_within_row = 0
    for raw in series:
        if not raw:
            continue
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if len(tokens) != len(set(tokens)):
            duplicate_within_row += 1
        for token in tokens:
            if token.isdigit():
                counts[token] += 1
            else:
                malformed += 1
    return counts, malformed, duplicate_within_row


def label_summary(
    frame: pd.DataFrame,
    code_column: str,
    patient_column: str,
    dictionary: pd.DataFrame,
) -> tuple[dict, dict[str, pd.Series]]:
    known = set(dictionary["codes"])
    tokens, malformed, duplicate_rows = token_counter(frame[code_column])
    invalid = {code: int(count) for code, count in sorted(tokens.items()) if code not in known}
    result: dict = {
        "nonempty_rows": int(frame[code_column].ne("").sum()),
        "empty_rows": int(frame[code_column].eq("").sum()),
        "malformed_token_occurrences": malformed,
        "rows_with_duplicate_code_tokens": duplicate_rows,
        "unknown_code_occurrences": invalid,
        "outcomes": {},
    }
    masks: dict[str, pd.Series] = {}
    for outcome, mapping in CODE_GROUPS.items():
        strict = code_mask(frame[code_column], mapping["strict"])
        core = code_mask(frame[code_column], mapping["core"])
        ambiguous = code_mask(frame[code_column], mapping["ambiguous"])
        excluded = code_mask(frame[code_column], mapping["excluded_comparison_or_fragment"])
        masks[outcome] = core
        result["outcomes"][outcome] = {
            "snomed_reference_only": mapping["snomed"],
            "strict_codes": mapping["strict"],
            "core_codes": mapping["core"],
            "ambiguous_codes": mapping["ambiguous"],
            "excluded_comparison_or_fragment_codes": mapping["excluded_comparison_or_fragment"],
            "strict_recordings": int(strict.sum()),
            "strict_patient_linked_recordings": int(
                (strict & frame[patient_column].ne("")).sum()
            ),
            "strict_patients": int(frame.loc[strict & frame[patient_column].ne(""), patient_column].nunique()),
            "core_recordings": int(core.sum()),
            "core_patient_linked_recordings": int(
                (core & frame[patient_column].ne("")).sum()
            ),
            "core_patients": int(frame.loc[core & frame[patient_column].ne(""), patient_column].nunique()),
            "ambiguous_recordings": int(ambiguous.sum()),
            "ambiguous_patient_linked_recordings": int(
                (ambiguous & frame[patient_column].ne("")).sum()
            ),
            "ambiguous_patients": int(frame.loc[ambiguous & frame[patient_column].ne(""), patient_column].nunique()),
            "excluded_comparison_or_fragment_recordings": int(excluded.sum()),
            "per_code_occurrences": {
                str(code): int(tokens.get(str(code), 0))
                for code in mapping["strict"]
                + [c for c in mapping["core"] if c not in mapping["strict"]]
                + mapping["ambiguous"]
                + mapping["excluded_comparison_or_fragment"]
            },
        }

    bits = pd.DataFrame({name: mask.astype(np.uint8) for name, mask in masks.items()})
    bit_strings = bits.astype(str).agg("".join, axis=1)
    result["core_recording_label_patterns_TWA_SB_STach"] = {
        key: int(value) for key, value in bit_strings.value_counts().sort_index().items()
    }
    patient_bits = bits.assign(_patient=frame[patient_column]).loc[frame[patient_column].ne("")]
    patient_bits = patient_bits.groupby("_patient", sort=False)[list(masks)].max()
    patient_strings = patient_bits.astype(str).agg("".join, axis=1)
    result["core_patient_label_patterns_TWA_SB_STach"] = {
        key: int(value) for key, value in patient_strings.value_counts().sort_index().items()
    }
    return result, masks


def agreement(left: pd.Series, right: pd.Series) -> dict:
    return {
        "both_positive": int((left & right).sum()),
        "left_only": int((left & ~right).sum()),
        "right_only": int((~left & right).sum()),
        "both_negative": int((~left & ~right).sum()),
        "agreement_fraction": float((left == right).mean()),
    }


def main() -> None:
    for path in PATHS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    dictionary = pd.read_csv(PATHS["dictionary"], dtype=str, keep_default_na=False)
    acquisition = pd.read_csv(PATHS["acquisition"], dtype=str, keep_default_na=False)
    v24 = pd.read_csv(PATHS["v24"], dtype=str, keep_default_na=False)
    metadata = pd.read_csv(PATHS["metadata"], dtype=str, keep_default_na=False)

    file_info = {
        name: {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in PATHS.items()
    }

    result: dict = {
        "version": "heedb_readiness_v1",
        "files": file_info,
        "schemas": {
            "dictionary": frame_summary(dictionary),
            "acquisition": frame_summary(acquisition),
            "v24": frame_summary(v24),
            "metadata": frame_summary(metadata),
        },
    }

    result["keys"] = {
        "metadata_FileName_duplicate_rows": int(metadata["FileName"].duplicated(keep=False).sum()),
        "metadata_FileID_duplicate_rows": int(metadata["FileID"].duplicated(keep=False).sum()),
        "metadata_BDSPPatientID_missing_rows": int(metadata["BDSPPatientID"].eq("").sum()),
        "metadata_BDSPPatientID_unique_nonempty": int(
            metadata.loc[metadata["BDSPPatientID"].ne(""), "BDSPPatientID"].nunique()
        ),
        "acquisition_FileName_duplicate_rows": int(acquisition["FileName"].duplicated(keep=False).sum()),
        "v24_FileName_duplicate_rows": int(v24["FileName"].duplicated(keep=False).sum()),
    }

    meta_files = set(metadata["FileName"])
    acquisition_files = set(acquisition["FileName"])
    v24_files = set(v24["FileName"])
    result["joins"] = {
        "acquisition_to_metadata_matched": len(acquisition_files & meta_files),
        "acquisition_without_metadata": len(acquisition_files - meta_files),
        "metadata_without_acquisition": len(meta_files - acquisition_files),
        "v24_to_metadata_matched": len(v24_files & meta_files),
        "v24_without_metadata": len(v24_files - meta_files),
        "metadata_without_v24": len(meta_files - v24_files),
        "v24_to_acquisition_matched": len(v24_files & acquisition_files),
        "v24_without_acquisition": len(v24_files - acquisition_files),
        "acquisition_without_v24": len(acquisition_files - v24_files),
    }

    patient_counts = metadata.loc[metadata["BDSPPatientID"].ne("")].groupby("BDSPPatientID").size()
    result["patient_recordings"] = {
        "patients": int(len(patient_counts)),
        "recordings": int(patient_counts.sum()),
        "patients_with_multiple_recordings": int((patient_counts > 1).sum()),
        "max_recordings_per_patient": int(patient_counts.max()),
        "recordings_per_patient_quantiles": {
            str(q): float(patient_counts.quantile(q)) for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        },
    }

    patient_lookup = metadata.set_index("FileName")["BDSPPatientID"]
    acquisition = acquisition.assign(BDSPPatientID=acquisition["FileName"].map(patient_lookup).fillna(""))
    v24 = v24.assign(BDSPPatientID=v24["FileName"].map(patient_lookup).fillna(""))

    software_summary, software_masks = label_summary(
        acquisition, "codes_software", "BDSPPatientID", dictionary
    )
    physician_summary, physician_masks = label_summary(
        acquisition, "codes_physician", "BDSPPatientID", dictionary
    )
    v24_summary, v24_masks = label_summary(v24, "codes", "BDSPPatientID", dictionary)
    result["labels"] = {
        "software_at_acquisition": software_summary,
        "physician_overread_at_acquisition": physician_summary,
        "v24_provenance_unspecified_by_schema": v24_summary,
    }
    result["software_vs_physician_core_recording_agreement"] = {
        outcome: agreement(software_masks[outcome], physician_masks[outcome])
        for outcome in CODE_GROUPS
    }

    overlap = v24[["FileName", "codes"]].merge(
        acquisition[["FileName", "codes_software", "codes_physician"]],
        on="FileName",
        how="inner",
        validate="one_to_one",
    )
    result["v24_comparison"] = {"overlap_recordings": int(len(overlap)), "core_outcome_agreement": {}}
    for outcome, mapping in CODE_GROUPS.items():
        vmask = code_mask(overlap["codes"], mapping["core"])
        smask = code_mask(overlap["codes_software"], mapping["core"])
        pmask = code_mask(overlap["codes_physician"], mapping["core"])
        result["v24_comparison"]["core_outcome_agreement"][outcome] = {
            "v24_vs_software": agreement(vmask, smask),
            "v24_vs_physician": agreement(vmask, pmask),
        }

    jaccard = pd.to_numeric(acquisition["jaccard_index"], errors="coerce")
    result["reported_code_jaccard"] = {
        "numeric_rows": int(jaccard.notna().sum()),
        "missing_or_nonnumeric_rows": int(jaccard.isna().sum()),
        "mean": float(jaccard.mean()),
        "median": float(jaccard.median()),
        "minimum": float(jaccard.min()),
        "maximum": float(jaccard.max()),
        "exactly_one": int(jaccard.eq(1).sum()),
    }

    # The mapping candidate rows contain no patient-level fields.
    relevant_codes = sorted(
        {
            code
            for mapping in CODE_GROUPS.values()
            for group in ("strict", "core", "ambiguous", "excluded_comparison_or_fragment")
            for code in mapping[group]
        }
    )
    relevant_dictionary = dictionary[dictionary["codes"].isin({str(c) for c in relevant_codes})]
    result["mapping_dictionary_rows"] = relevant_dictionary.to_dict(orient="records")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "rows": {name: result["schemas"][name]["rows"] for name in result["schemas"]},
        "joins": result["joins"],
        "patients": result["patient_recordings"]["patients"],
        "labels": {
            source: {
                outcome: {
                    "strict_recordings": detail["strict_recordings"],
                    "strict_patients": detail["strict_patients"],
                    "core_recordings": detail["core_recordings"],
                    "core_patients": detail["core_patients"],
                    "ambiguous_recordings": detail["ambiguous_recordings"],
                }
                for outcome, detail in summary["outcomes"].items()
            }
            for source, summary in result["labels"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
