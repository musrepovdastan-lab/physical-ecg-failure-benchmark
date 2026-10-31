import numpy as np

from src.ecg_physics import (
    DISPLAY_LEADS,
    INDEPENDENT_LEADS,
    derive_display,
    fault_unavailability,
    limb_terminal_permutation,
    named_reversal,
)


def sample():
    return np.random.default_rng(7).normal(size=(3, 8, 101))


def test_einthoven_and_goldberger_identities():
    y = derive_display(sample())
    i, ii, iii, avr, avl, avf = [y[:, index, :] for index in range(6)]
    np.testing.assert_allclose(ii, i + iii)
    np.testing.assert_allclose(avr, -(i + ii) / 2)
    np.testing.assert_allclose(avl, i - ii / 2)
    np.testing.assert_allclose(avf, ii - i / 2)
    np.testing.assert_allclose(avr + avl + avf, np.zeros_like(i), atol=1e-12)


def test_ra_la_reversal_equations_and_chest_invariance():
    y = derive_display(named_reversal(sample(), "RA_LA_reversal"))
    original = derive_display(sample())  # shape check only; equations below use same input
    x = sample()
    before, after = derive_display(x), derive_display(named_reversal(x, "RA_LA_reversal"))
    np.testing.assert_allclose(after[:, 0], -before[:, 0])
    np.testing.assert_allclose(after[:, 1], before[:, 2])
    np.testing.assert_allclose(after[:, 2], before[:, 1])
    np.testing.assert_allclose(after[:, 3], before[:, 4])
    np.testing.assert_allclose(after[:, 4], before[:, 3])
    np.testing.assert_allclose(after[:, 5], before[:, 5])
    np.testing.assert_allclose(after[:, 6:], before[:, 6:])


def test_other_reversals_preserve_electrical_identities():
    x = sample()
    for fault in ("RA_LL_reversal", "LA_LL_reversal"):
        y = derive_display(named_reversal(x, fault))
        np.testing.assert_allclose(y[:, 1], y[:, 0] + y[:, 2])
        np.testing.assert_allclose(y[:, 3] + y[:, 4] + y[:, 5], 0, atol=1e-12)


def test_reversible_terminal_permutations_restore_original_waveform():
    x = sample()
    inverses = {
        "RA_LA_reversal": (1, 0, 2),
        "RA_LL_reversal": (2, 1, 0),
        "LA_LL_reversal": (0, 2, 1),
    }
    for fault, inverse in inverses.items():
        restored = limb_terminal_permutation(named_reversal(x, fault), inverse)
        np.testing.assert_allclose(restored, x, atol=1e-12)
        np.testing.assert_allclose(derive_display(restored), derive_display(x), atol=1e-12)


def test_each_predicted_swap_class_is_its_exact_inverse_for_independent_and_displayed_signals():
    # Multiples of six keep every WCT division and terminal difference exact.
    time = np.arange(32, dtype=np.float64)
    base = np.stack((6 * time, 12 * time, *(18 * lead + 6 * time for lead in range(1, 7))), axis=0)[None]
    predicted_classes = {
        1: "RA_LA_reversal",
        2: "RA_LL_reversal",
        3: "LA_LL_reversal",
    }
    for predicted_class, reversal in predicted_classes.items():
        assert predicted_class in (1, 2, 3)
        faulted = named_reversal(base, reversal)
        restored = named_reversal(faulted, reversal)
        np.testing.assert_array_equal(restored, base)
        np.testing.assert_array_equal(derive_display(restored), derive_display(base))


def test_ra_ll_reversal_changes_exactly_five_displayed_leads_and_preserves_avl_and_chest():
    # Non-degenerate terminal signals make each changed-lead assertion meaningful.
    t = np.linspace(-1.0, 1.0, 101)
    x = np.stack((1.7 + 0.3 * t, -0.8 + 0.6 * t, *(0.1 * k + t**2 for k in range(1, 7))), axis=0)[None]
    before = derive_display(x)
    after = derive_display(named_reversal(x, "RA_LL_reversal"))
    changed = ("I", "II", "III", "aVR", "aVF")
    unchanged = ("aVL", "V1", "V2", "V3", "V4", "V5", "V6")
    index = {name: i for i, name in enumerate(DISPLAY_LEADS)}
    for name in changed:
        assert not np.allclose(after[:, index[name]], before[:, index[name]])
    for name in unchanged:
        np.testing.assert_allclose(after[:, index[name]], before[:, index[name]], atol=1e-12)
    np.testing.assert_allclose(after[:, index["I"]], -before[:, index["III"]])
    np.testing.assert_allclose(after[:, index["II"]], -before[:, index["II"]])
    np.testing.assert_allclose(after[:, index["III"]], -before[:, index["I"]])
    np.testing.assert_allclose(after[:, index["aVR"]], before[:, index["aVF"]])
    np.testing.assert_allclose(after[:, index["aVF"]], before[:, index["aVR"]])


def test_each_implemented_unavailability_pattern_is_exact_and_nan_only_where_declared():
    x = sample()
    singles = tuple(f"precordial_electrode_{v}_disconnection" for v in ("V1", "V2", "V3", "V4", "V5", "V6"))
    for fault in singles:
        out, mask = fault_unavailability(x, fault)
        assert mask.sum() == 1
        assert np.isnan(out[:, mask]).all()
        assert np.isfinite(out[:, ~mask]).all()
    out, mask = fault_unavailability(x, "coherent_precordial_failure")
    assert tuple(np.array(INDEPENDENT_LEADS)[mask]) == ("V1", "V2", "V3", "V4", "V5", "V6")
    assert np.isnan(out[:, mask]).all()
    for terminal in ("RA", "LA", "LL"):
        out, mask = fault_unavailability(x, f"{terminal}_disconnection_or_WCT_failure")
        assert mask.all() and np.isnan(out).all()
