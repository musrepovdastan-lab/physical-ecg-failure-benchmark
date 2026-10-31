"""Electrical and preprocessing checks for the ECGFounder baseline."""
import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, filtfilt, iirnotch, medfilt

from src.ecg_physics import derive_display, named_reversal


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCE = ROOT / "outputs/ecgfounder_baseline_v1/official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23"
requires_official_source = pytest.mark.skipif(
    not all((OFFICIAL_SOURCE / name).is_file() for name in ("net1d.py", "util.py")),
    reason="Install the pinned ECGFounder source; see docs/REPRODUCING.md",
)


def test_ra_ll_lead_equations_and_chest_invariance():
    rng = np.random.default_rng(120)
    independent = rng.normal(size=(8, 5000))
    display = derive_display(independent)
    changed = derive_display(named_reversal(independent, "RA_LL_reversal"))
    i, ii = display[0], display[1]
    np.testing.assert_allclose(changed[0], i - ii, rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed[1], -ii, rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed[2], -i, rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed[3], display[5], rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed[4], display[4], rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed[5], display[3], rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed[6:], display[6:], rtol=0, atol=1e-12)
    np.testing.assert_allclose(derive_display(named_reversal(
        named_reversal(independent, "RA_LL_reversal"), "RA_LL_reversal")), display,
        rtol=0, atol=1e-12)


@requires_official_source
def test_filter_matches_pinned_official_function():
    # Compile only the authors' filter_bandpass function to avoid util.py's
    # unrelated optional plotting and HDF5 imports.
    source = (ROOT / "outputs/ecgfounder_baseline_v1/official/"
              "ECGFounder-04edac702b61c91face519774ddcc0cd712fef23/util.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "filter_bandpass")
    module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
    env = {"np": np, "iirnotch": iirnotch, "filtfilt": filtfilt, "butter": butter, "medfilt": medfilt}
    exec(compile(module, str(source), "exec"), env)
    from src.ecgfounder_baseline_v1 import official_filter, preprocess

    x = np.random.default_rng(125).normal(size=(12, 5000)).astype(np.float64)
    expected = env["filter_bandpass"](x, 500)
    np.testing.assert_array_equal(official_filter(x), expected)
    z = preprocess(x)
    np.testing.assert_allclose(z, ((expected - expected.mean()) / (expected.std() + 1e-8)).astype(np.float32), rtol=0, atol=0)


@requires_official_source
def test_reversal_precedes_nonlinear_preprocessing():
    from src.ecgfounder_baseline_v1 import preprocess, reverse_ra_ll

    rng = np.random.default_rng(126)
    x = derive_display(rng.normal(size=(8, 5000)))
    expected = preprocess(derive_display(named_reversal(
        np.concatenate((x[:2], x[6:]), axis=0), "RA_LL_reversal")))
    np.testing.assert_array_equal(preprocess(reverse_ra_ll(x)), expected)
