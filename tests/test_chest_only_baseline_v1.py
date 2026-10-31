import numpy as np
import torch

from src.chest_only_baseline_v1 import (
    ChestOnlyResNetSmall,
    ResNet12,
    ece15,
    metrics,
    normalized_chest,
    physical_reversal_chest,
    stats,
    thresholds,
)


def test_architecture_changes_only_input_stem():
    six = ChestOnlyResNetSmall()
    twelve = ResNet12()
    assert six.s[0].weight.shape == (32, 6, 7)
    assert twelve.s[0].weight.shape == (32, 12, 7)
    assert set(six.state_dict()) == set(twelve.state_dict())
    assert sum(p.numel() for p in six.parameters()) - sum(p.numel() for p in twelve.parameters()) == -1344
    with torch.no_grad():
        assert six(torch.zeros(2, 6, 257)).shape == (2, 3)


def test_all_physical_reversals_preserve_chest_signals_exactly():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(12, 1280)).astype(np.float32)
    for reversal in ("RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal"):
        assert np.array_equal(physical_reversal_chest(x, reversal), x[6:12])


def test_normalized_inference_uses_only_train_chest_stats(monkeypatch):
    x = np.arange(12 * 30, dtype=np.float32).reshape(12, 30)
    monkeypatch.setattr("src.chest_only_baseline_v1.raw12", lambda dataset, record: x.copy())
    mean = np.arange(6, dtype=np.float32)
    std = np.arange(1, 7, dtype=np.float32)
    expected = (x[6:12] - mean[:, None]) / std[:, None]
    assert np.array_equal(normalized_chest("Chapman", "r1", mean, std), expected)
    for reversal in ("RA_LA_reversal", "RA_LL_reversal", "LA_LL_reversal"):
        assert np.array_equal(normalized_chest("Chapman", "r1", mean, std, reversal), expected)


def test_stats_are_per_channel_and_population_moments(monkeypatch):
    a = np.arange(12 * 4, dtype=np.float32).reshape(12, 4)
    b = a + 2
    seq = iter((a, b))
    monkeypatch.setattr("src.chest_only_baseline_v1.raw12", lambda dataset, record: next(seq))
    mean, std = stats("Chapman", ["a", "b"])
    joined = np.concatenate((a[6:12], b[6:12]), axis=1)
    assert np.allclose(mean, joined.mean(1), rtol=0, atol=1e-6)
    assert np.allclose(std, joined.std(1), rtol=0, atol=1e-6)


def test_validation_thresholds_metrics_and_ece_match_frozen_definition():
    y = np.asarray([[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    p = np.asarray([[.05, .1, .2], [.2, .7, .3], [.8, .3, .75], [.9, .8, .95]], dtype=np.float32)
    t = thresholds(y, p)
    assert np.allclose(t, [.185, .28, .29])
    rows = metrics(y, p, t)
    assert [r["endpoint"] for r in rows] == ["TWA", "SB", "STach"]
    assert all(0 <= r["ece_15"] <= 1 and 0 <= r["brier"] <= 1 for r in rows)
    assert rows[0]["sensitivity"] == 1.0
    assert np.isclose(ece15(y[:, 0], p[:, 0]), rows[0]["ece_15"])
