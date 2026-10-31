"""Known v1 adapter failures and exact RA-LL algebra for v2."""
from pathlib import Path

import numpy as np
import pytest
import wfdb

OFFICIAL_SOURCE = Path(__file__).resolve().parents[1] / "outputs/ecgfounder_baseline_v1/official/ECGFounder-04edac702b61c91face519774ddcc0cd712fef23"
if not (OFFICIAL_SOURCE / "net1d.py").is_file():
    pytest.skip("Install the pinned ECGFounder source; see docs/REPRODUCING.md", allow_module_level=True)

from src.ecgfounder_baseline_v2 import physical_signal, preprocess, record_path, reverse_ra_ll
from src.ecg_physics import derive_display


def test_missing_physical_samples_replaced_before_reversal_and_filter():
    path = record_path("Ningbo", "g1/JS10767")
    x, _ = wfdb.rdsamp(str(path))
    assert int(np.isnan(x).sum()) == 4
    expected = np.nan_to_num(x, nan=0.0).T
    actual = physical_signal("Ningbo", "g1/JS10767")
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(preprocess(actual)).all()
    assert np.isfinite(preprocess(reverse_ra_ll(actual))).all()


def test_moved_canonical_path_resolves():
    path = record_path("Ningbo", "g11/JS21646")
    assert path.as_posix().endswith("/ningbo/g12/JS21646")
    assert physical_signal("Ningbo", "g11/JS21646").shape == (12, 5000)


def test_canonical_header_can_have_distinct_stem():
    path = record_path("Ningbo", "g13/JS23074")
    assert path.as_posix().endswith("/ningbo/g13/S23074")
    assert physical_signal("Ningbo", "g13/JS23074").shape == (12, 5000)


def test_exact_ra_ll_equations():
    x = physical_signal("Ningbo", "g1/JS10767")
    changed = reverse_ra_ll(x)
    i, ii = x[0], x[1]
    np.testing.assert_allclose(changed[0], i - ii, rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed[1], -ii, rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed[2], -i, rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed[6:], x[6:], rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed[3], -(changed[0] + changed[1]) / 2, rtol=0, atol=1e-12)
    independent = np.concatenate((changed[:2], changed[6:]), axis=0)
    np.testing.assert_allclose(derive_display(independent), changed, rtol=0, atol=1e-12)
