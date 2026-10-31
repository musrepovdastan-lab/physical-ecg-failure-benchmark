"""Exact terminal algebra for a conventional simultaneously recorded 12-lead ECG.

Input is the eight independent displayed measurements I, II, V1..V6.  The gauge is
RA=0, LA=I, LL=II.  This is an algebraic representation—not a model of skin-electrode
impedance, movement artefact, or an open-circuit device response.
"""
from __future__ import annotations

import numpy as np

INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
DISPLAY_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def _as_independent(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim < 2 or x.shape[-2] != 8:
        raise ValueError("expected (..., 8, time) independent leads in I, II, V1..V6 order")
    return x


def derive_display(independent: np.ndarray) -> np.ndarray:
    """Return I, II, III, aVR, aVL, aVF, V1..V6 using standard equations."""
    x = _as_independent(independent)
    i, ii = x[..., 0, :], x[..., 1, :]
    limb = np.stack((i, ii, ii - i, -(i + ii) / 2, i - ii / 2, ii - i / 2), axis=-2)
    return np.concatenate((limb, x[..., 2:, :]), axis=-2)


def terminal_potentials(independent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return RA, LA, LL and WCT in the RA=0 gauge."""
    x = _as_independent(independent)
    ra = np.zeros_like(x[..., 0, :])
    la, ll = x[..., 0, :], x[..., 1, :]
    return ra, la, ll, (ra + la + ll) / 3


def limb_terminal_permutation(independent: np.ndarray, permutation: tuple[int, int, int]) -> np.ndarray:
    """Simulate an exact limb *cable/terminal* permutation.

    permutation identifies the physical terminal connected to displayed slots (RA, LA, LL),
    using 0=RA, 1=LA, 2=LL. Chest leads are unchanged because WCT is permutation-invariant.
    This is valid for cable reversals, not for an electrode detachment.
    """
    if sorted(permutation) != [0, 1, 2]:
        raise ValueError("permutation must contain RA, LA, LL exactly once")
    x = _as_independent(independent)
    ra, la, ll, wct = terminal_potentials(x)
    physical = (ra, la, ll)
    slot_ra, slot_la, slot_ll = (physical[j] for j in permutation)
    new_i = slot_la - slot_ra
    new_ii = slot_ll - slot_ra
    # The physical chest potential is held fixed; WCT is mathematically invariant to reordering.
    chest_potentials = x[..., 2:, :] + wct[..., None, :]
    new_wct = (slot_ra + slot_la + slot_ll) / 3
    new_chest = chest_potentials - new_wct[..., None, :]
    return np.concatenate((np.stack((new_i, new_ii), axis=-2), new_chest), axis=-2)


def unavailable_independent(independent: np.ndarray, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Represent an unobservable independent measurement as NaN, never fabricated zero signal.

    The returned boolean mask is True where the measurement is unavailable.  A Phase-1
    inference adapter may encode this mask only after a separately validated rule is frozen.
    """
    x = _as_independent(independent).copy()
    mask = np.zeros(x.shape[-2], dtype=bool)
    for name in names:
        try:
            mask[INDEPENDENT_LEADS.index(name)] = True
        except ValueError as exc:
            raise ValueError(f"not an independent lead: {name}") from exc
    x[..., mask, :] = np.nan
    return x, mask


def fault_unavailability(independent: np.ndarray, fault: str) -> tuple[np.ndarray, np.ndarray]:
    """Exact observable/unobservable consequence classes, without an impedance heuristic."""
    mapping = {
        "precordial_electrode_V1_disconnection": ("V1",),
        "precordial_electrode_V2_disconnection": ("V2",),
        "precordial_electrode_V3_disconnection": ("V3",),
        "precordial_electrode_V4_disconnection": ("V4",),
        "precordial_electrode_V5_disconnection": ("V5",),
        "precordial_electrode_V6_disconnection": ("V6",),
        "coherent_precordial_failure": ("V1", "V2", "V3", "V4", "V5", "V6"),
        # A limb terminal loss makes I/II and the WCT-referenced chest leads unidentified.
        "RA_disconnection_or_WCT_failure": INDEPENDENT_LEADS,
        "LA_disconnection_or_WCT_failure": INDEPENDENT_LEADS,
        "LL_disconnection_or_WCT_failure": INDEPENDENT_LEADS,
    }
    try:
        return unavailable_independent(independent, mapping[fault])
    except KeyError as exc:
        raise ValueError(f"unknown non-identifiable fault: {fault}") from exc


def named_reversal(independent: np.ndarray, fault: str) -> np.ndarray:
    """Apply one physically exact limb cable reversal."""
    mapping = {
        "RA_LA_reversal": (1, 0, 2),
        "RA_LL_reversal": (2, 1, 0),
        "LA_LL_reversal": (0, 2, 1),
    }
    try:
        return limb_terminal_permutation(independent, mapping[fault])
    except KeyError as exc:
        raise ValueError(f"unknown reversible fault: {fault}") from exc
