import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.architecture_m3_recovery_v2 import (
    EXACT,
    MASKS,
    apply_mask_raw,
    infer_batches,
    prepare_masked_input,
    weighted_ap_resamples,
)


def test_mask_means_remove_exact_five_and_preserve_other_seven_with_zero_sentinel():
    x = np.arange(2 * 12 * 17, dtype=np.float32).reshape(2, 12, 17) + 0.25
    mean = np.arange(12, dtype=np.float32) * 2.0 + 10.0
    std = np.arange(12, dtype=np.float32) * 0.1 + 1.0
    masked = apply_mask_raw(x, mean, EXACT)
    kept = [i for i in range(12) if i not in EXACT]
    assert np.array_equal(masked[:, list(EXACT)], np.broadcast_to(mean[list(EXACT)][None, :, None], masked[:, list(EXACT)].shape))
    assert masked[:, kept].tobytes() == x[:, kept].tobytes()
    normalized = prepare_masked_input(x, mean, std, EXACT)
    assert np.array_equal(normalized[:, list(EXACT)], np.zeros_like(normalized[:, list(EXACT)]))


def test_exact_mask_is_unique_panel_column_and_real_inference_receives_expected_tensor():
    assert len(MASKS) == 792 and len(set(MASKS)) == 792
    assert MASKS.index(EXACT) == 1  # combinations in lexicographic order
    x = np.arange(3 * 12 * 20, dtype=np.float32).reshape(3, 12, 20) + 0.5
    mean = np.arange(12, dtype=np.float32) + 1
    std = np.arange(12, dtype=np.float32) * 0.2 + 1
    expected = prepare_masked_input(x, mean, std, EXACT)

    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, z):
            self.seen.append(z.detach().cpu().numpy().copy())
            return torch.zeros((len(z), 3), device=z.device)

    model = Capture()
    scores = infer_batches(model, expected, "cpu", batch_size=2, amp=False)
    observed = np.concatenate(model.seen, axis=0)
    assert np.array_equal(observed, expected)
    assert scores.shape == (3, 3) and np.isfinite(scores).all()


def test_weighted_bootstrap_ap_matches_sklearn_for_ties_and_repeated_records():
    rng = np.random.default_rng(20261017)
    y = rng.integers(0, 2, size=31, dtype=np.uint8)
    scores = np.round(rng.random((4, len(y))), 1).astype(np.float32)
    indices = rng.integers(0, len(y), size=(11, len(y)), dtype=np.int64)
    counts = np.stack([np.bincount(row, minlength=len(y)) for row in indices]).astype(np.uint16)
    got = weighted_ap_resamples(y, scores, counts)
    expected = np.asarray([[average_precision_score(y[row], scores[c, row]) for row in indices]
                           for c in range(len(scores))])
    assert np.allclose(got, expected, rtol=0, atol=1e-7)
