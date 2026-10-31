import itertools

import numpy as np
import torch

from src.mask_semantics_recovery_v2 import (
    EXACT_INDICES,
    LEADS,
    MASKS,
    apply_mask_raw,
    infer_batches,
    prepare_masked_input,
    weighted_ap_resamples,
)


def sample_arrays(batch=1):
    x = np.arange(batch * 12 * 7, dtype=np.float32).reshape(batch, 12, 7) + 0.25
    mean = np.arange(12, dtype=np.float32) * 3 + 11.0
    std = np.arange(12, dtype=np.float32) * 0.2 + 1.25
    return x, mean, std


def test_working_mask_path_single_example_and_batch_preserves_unselected_bytes():
    x, mean, std = sample_arrays(batch=2)
    mask = EXACT_INDICES
    for raw in (x[0], x):
        masked = apply_mask_raw(raw, mean, mask)
        axis = 0 if raw.ndim == 2 else 1
        selected = np.asarray(mask)
        retained = np.asarray([i for i in range(12) if i not in mask])
        if axis == 0:
            assert np.array_equal(masked[selected], np.broadcast_to(mean[selected, None], masked[selected].shape))
            assert masked[retained].tobytes() == raw[retained].tobytes()
        else:
            assert np.array_equal(masked[:, selected], np.broadcast_to(mean[selected][None, :, None], masked[:, selected].shape))
            assert masked[:, retained].tobytes() == raw[:, retained].tobytes()
        normalized = prepare_masked_input(raw, mean, std, mask)
        if axis == 0:
            assert np.array_equal(normalized[selected], np.zeros_like(normalized[selected]))
        else:
            assert np.array_equal(normalized[:, selected], np.zeros_like(normalized[:, selected]))


def test_exact_mask_names_and_all_792_masks_are_unique():
    assert [LEADS[i] for i in EXACT_INDICES] == ["I", "II", "III", "aVR", "aVF"]
    assert len(MASKS) == 792
    assert len(set(MASKS)) == 792
    assert all(len(x) == 5 and len(set(x)) == 5 for x in MASKS)
    assert MASKS[0] == tuple(itertools.combinations(range(12), 5).__iter__().__next__())


class CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().cpu().numpy().copy())
        return torch.stack((x[:, 0].mean(1), x[:, 1].mean(1), x[:, 2].mean(1)), dim=1)


def test_actual_inference_adapter_receives_expected_normalized_mask_tensor_single_and_batch():
    x, mean, std = sample_arrays(batch=3)
    for raw in (x[0:1], x):
        model = CaptureModel()
        expected = prepare_masked_input(raw, mean, std, EXACT_INDICES)
        infer_batches(model, expected, "cpu", batch_size=2, amp=False)
        observed = np.concatenate(model.seen, axis=0)
        assert observed.shape == expected.shape
        assert np.array_equal(observed, expected)


def test_weighted_bootstrap_ap_matches_sklearn_with_repeated_samples_and_ties():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(123)
    y = rng.integers(0, 2, size=23, dtype=np.uint8)
    # Include ties deliberately: weighted AP must retain sklearn's threshold semantics.
    scores = np.round(rng.random((4, 23)), 1).astype(np.float32)
    indices = rng.integers(0, len(y), size=(9, len(y)), dtype=np.int64)
    counts = np.stack([np.bincount(row, minlength=len(y)) for row in indices]).astype(np.uint16)
    actual = weighted_ap_resamples(y, scores, counts)
    expected = np.asarray([[average_precision_score(y[row], scores[c, row]) for row in indices] for c in range(len(scores))])
    assert np.allclose(actual, expected, rtol=0, atol=1e-7)
