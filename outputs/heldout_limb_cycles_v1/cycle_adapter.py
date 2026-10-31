"""Versioned adapter for the two held-out limb-terminal cycles.

The historical source signals are 12 displayed leads. The project physics API
accepts the eight independent measurements I, II, V1-V6, so this adapter extracts
those measurements, applies the specified physical-slot permutation, then derives
the displayed limb leads. The historical clean control remains the unmodified
12-channel source array; it is never rewritten to match the derived cycles.
"""
from __future__ import annotations

import numpy as np

from src.ecg_physics import derive_display, limb_terminal_permutation

CHANNELS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
INDEPENDENT_INDEX = (0, 1, 6, 7, 8, 9, 10, 11)
CYCLES = {
    "cycle_slots_120": (1, 2, 0),
    "cycle_slots_201": (2, 0, 1),
}
KNOWN_SWAPS = {
    "RA_LA_reversal": (1, 0, 2),
    "RA_LL_reversal": (2, 1, 0),
    "LA_LL_reversal": (0, 2, 1),
}


def independent_from_display(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.shape[-2] != 12:
        raise ValueError(f"expected 12 displayed channels, got {x.shape}")
    return x[..., INDEPENDENT_INDEX, :]


def transform(x: np.ndarray, condition: str) -> np.ndarray:
    """Return 12 displayed leads after a physical terminal-slot permutation."""
    try:
        permutation = CYCLES[condition]
    except KeyError as exc:
        raise ValueError(f"unknown held-out cycle: {condition}") from exc
    x = np.asarray(x)
    if x.shape[-2] != 12:
        raise ValueError(f"expected 12 displayed channels, got {x.shape}")
    changed = limb_terminal_permutation(independent_from_display(x), permutation)
    return derive_display(changed).astype(np.float32, copy=False)


def historical_ra_ll(x: np.ndarray) -> np.ndarray:
    """Match the already-used project RA-LL path: extract I, II, V1-V6 then derive."""
    x = np.asarray(x)
    if x.shape[-2] != 12:
        raise ValueError(f"expected 12 displayed channels, got {x.shape}")
    return derive_display(limb_terminal_permutation(independent_from_display(x), KNOWN_SWAPS["RA_LL_reversal"])).astype(np.float32, copy=False)


def apply_permutation_to_independent(x: np.ndarray, permutation: tuple[int, int, int]) -> np.ndarray:
    return limb_terminal_permutation(x, permutation)


def expected_explicit(x: np.ndarray, condition: str) -> np.ndarray:
    """Independent displayed-lead formulas, used only for the engineering audit."""
    x = np.asarray(x)
    i, ii = x[..., 0, :], x[..., 1, :]
    chest = x[..., 6:12, :]
    if condition == "cycle_slots_120":
        limb = np.stack((ii-i, -i, -ii, i-ii/2, ii-i/2, -(i+ii)/2), axis=-2)
    elif condition == "cycle_slots_201":
        limb = np.stack((-ii, -ii+i, i, ii-i/2, -(i+ii)/2, i-ii/2), axis=-2)
    else:
        raise ValueError(condition)
    # Keep the calculation dtype so the independent equation audit does not
    # compare float64 terminal algebra against prematurely rounded float32.
    return np.concatenate((limb, chest), axis=-2)
