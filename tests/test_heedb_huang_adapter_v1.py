import numpy as np
import pytest

from src.ecg_physics import derive_display, named_reversal
from src.heedb_huang_adapter_v1 import (
    corrected_batch, detector_input, diagnostic_input, electrical_reversal, electrical_reversal_batch, predicted_inverse,
)
from src.huang_detector_infer_v2 import preprocess_one


def waveform():
    t = np.arange(5000, dtype=np.float32) / 500
    i = 300 * np.sin(2 * np.pi * 1.3 * t) + 35 * np.sin(2 * np.pi * 18 * t)
    ii = 550 * np.sin(2 * np.pi * 1.3 * t + .7)
    chest = np.stack([100 * (k + 1) * np.cos(2 * np.pi * 1.1 * t + k / 4) for k in range(6)])
    return derive_display(np.concatenate((np.stack((i, ii)), chest))).astype(np.float32)


def test_published_detector_preprocessing_exact_lead_order_units_and_rate():
    x = waveform()
    actual = detector_input(x)
    assert actual.shape == (1200, 3)
    np.testing.assert_array_equal(actual, preprocess_one(x))
    np.testing.assert_allclose(actual, detector_input(x / 1000), atol=2e-5)
    for channel, source in enumerate((2, 3, 11)):
        only_one = x.copy()
        only_one[source] = x[source] + 100
        changed = detector_input(only_one)
        np.testing.assert_allclose(changed[:, channel], actual[:, channel], atol=2e-5)
        for other in set(range(3)) - {channel}:
            np.testing.assert_array_equal(changed[:, other], actual[:, other])


def test_exact_ra_ll_all_twelve_and_augmented_swap():
    x = waveform()
    reversed_x = electrical_reversal(x, "RA_LL_reversal")
    independent = np.concatenate((x[:2], x[6:]))
    expected = derive_display(named_reversal(independent, "RA_LL_reversal"))
    np.testing.assert_allclose(reversed_x, expected, rtol=0, atol=1e-4)
    np.testing.assert_allclose(reversed_x[3], x[5], rtol=0, atol=1e-4)  # aVR' = aVF
    np.testing.assert_allclose(reversed_x[5], x[3], rtol=0, atol=1e-4)  # aVF' = aVR
    np.testing.assert_allclose(reversed_x[6:], x[6:], rtol=0, atol=1e-4)
    np.testing.assert_allclose(electrical_reversal(reversed_x, "RA_LL_reversal"), x, rtol=0, atol=1e-4)
    np.testing.assert_allclose(electrical_reversal_batch(np.stack((x, x)), "RA_LL_reversal"),
                               np.stack((reversed_x, reversed_x)), rtol=0, atol=1e-4)


def test_argmax_inverse_and_exact_diagnostic_time_axis():
    x = waveform()
    mean, std = np.zeros(12), np.ones(12)
    assert diagnostic_input(x, mean, std).shape == (12, 1000)
    np.testing.assert_array_equal(diagnostic_input(x, mean, std), x[:, ::5])
    np.testing.assert_array_equal(predicted_inverse(x, 0), x)
    for code, name in ((1, "RA_LA_reversal"), (2, "RA_LL_reversal"), (3, "LA_LL_reversal")):
        np.testing.assert_allclose(predicted_inverse(x, code), electrical_reversal(x, name), atol=1e-4)
    batch = np.stack((x, x, x, x))
    corrected = corrected_batch(batch, np.arange(4))
    for code in range(4):
        np.testing.assert_allclose(corrected[code], predicted_inverse(x, code), atol=1e-4)
    with pytest.raises(ValueError):
        predicted_inverse(x, 4)
    with pytest.raises(ValueError):
        detector_input(x[:, ::5])
