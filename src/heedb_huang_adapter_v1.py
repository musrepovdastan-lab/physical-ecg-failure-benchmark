"""Outcome-blind HEEDB-to-published-Huang adapter (no learned adaptation)."""
from __future__ import annotations

import numpy as np

from src.ecg_physics import DISPLAY_LEADS, derive_display, named_reversal
from src.huang_detector_infer_v2 import preprocess_one

HUANG_LEADS = ("III", "aVR", "V6")
HUANG_CLASSES = ("clean", "RA_LA", "RA_LL", "LA_LL")
CLASS_TO_REVERSAL = {
    1: "RA_LA_reversal",
    2: "RA_LL_reversal",
    3: "LA_LL_reversal",
}


def validate_raw(raw: np.ndarray) -> np.ndarray:
    """Frozen cache: canonical display leads, 10 seconds at 500 Hz, microvolts."""
    x = np.asarray(raw, dtype=np.float32)
    if x.shape != (12, 5000) or not np.isfinite(x).all():
        raise ValueError("expected finite (12,5000) canonical 500-Hz microvolt ECG")
    if DISPLAY_LEADS != ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"):
        raise RuntimeError("project canonical lead order changed")
    return x


def electrical_reversal(raw: np.ndarray, name: str) -> np.ndarray:
    x = validate_raw(raw)
    independent = np.concatenate((x[:2], x[6:]), axis=0)
    y = derive_display(named_reversal(independent, name)).astype(np.float32)
    if y.shape != x.shape or not np.isfinite(y).all():
        raise RuntimeError("invalid electrically reversed display")
    return y


def electrical_reversal_batch(raw: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (12, 5000) or not np.isfinite(x).all():
        raise ValueError("expected finite batch of canonical 500-Hz ECGs")
    independent = np.concatenate((x[:, :2], x[:, 6:]), axis=1)
    y = derive_display(named_reversal(independent, name)).astype(np.float32)
    if y.shape != x.shape or not np.isfinite(y).all():
        raise RuntimeError("invalid electrically reversed display batch")
    return y


def detector_input(raw: np.ndarray) -> np.ndarray:
    """Existing published Huang preprocessing: 0.5-35 Hz, 120 Hz, per-lead z.

    The source cache is in microvolts, while the publication's detector uses
    per-record/lead z-scores; no weight change or cohort-derived scaling occurs.
    """
    x = validate_raw(raw)
    prepared = preprocess_one(x)
    if prepared.shape != (1200, 3) or not np.isfinite(prepared).all():
        raise RuntimeError("invalid Huang input")
    return prepared


def predicted_inverse(raw: np.ndarray, predicted_class: int) -> np.ndarray:
    """Only the detector's argmax is consulted; never the simulated true fault."""
    x = validate_raw(raw)
    code = int(predicted_class)
    if code == 0:
        return x
    if code not in CLASS_TO_REVERSAL:
        raise ValueError("invalid detector class")
    return electrical_reversal(x, CLASS_TO_REVERSAL[code])


def corrected_batch(raw: np.ndarray, predicted_classes: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float32)
    codes = np.asarray(predicted_classes)
    if x.ndim != 3 or x.shape[1:] != (12, 5000) or codes.shape != (len(x),) or not np.isfinite(x).all() or not np.isin(codes, (0, 1, 2, 3)).all():
        raise ValueError("invalid batch or predicted detector classes")
    result = x.copy()
    for code, name in CLASS_TO_REVERSAL.items():
        mask = codes == code
        if mask.any():
            result[mask] = electrical_reversal_batch(x[mask], name)
    return result


def diagnostic_input(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = validate_raw(raw)[:, ::5]
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if x.shape != (12, 1000) or mean.shape != (12,) or std.shape != (12,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("invalid frozen Ningbo diagnostic input geometry")
    y = ((x - mean[:, None]) / std[:, None]).astype(np.float32)
    if not np.isfinite(y).all():
        raise ValueError("invalid normalized diagnostic input")
    return y
