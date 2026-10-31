"""Pre-inference engineering gates for corrected HEEDB input handling."""

import unittest

import numpy as np

from src.ecg_physics import derive_display, named_reversal
from src.heedb_external_corrected_v2 import exact_ra_ll, normalized_conditions, to_ningbo_time


class CorrectedInputTests(unittest.TestCase):
    def test_exact_ningbo_input_shape_and_sample_positions(self):
        raw = np.broadcast_to(np.arange(5000, dtype=np.float32), (2, 12, 5000)).copy()
        sampled = to_ningbo_time(raw)
        self.assertEqual(sampled.shape, (2, 12, 1000))
        np.testing.assert_array_equal(sampled[0, 0], np.arange(0, 5000, 5, dtype=np.float32))
        with self.assertRaises(ValueError):
            to_ningbo_time(np.zeros((2, 12, 1000), dtype=np.float32))
        with self.assertRaises(ValueError):
            to_ningbo_time(np.zeros((2, 11, 5000), dtype=np.float32))

    def test_all_twelve_reversed_leads_and_augmentation_signs(self):
        x = np.zeros((1, 12, 1000), dtype=np.float32)
        x[:, 0, :] = 2  # I
        x[:, 1, :] = 6  # II
        for lead in range(6, 12):
            x[:, lead, :] = 10 + lead
        observed = exact_ra_ll(x)
        self.assertEqual(observed.shape, (1, 12, 1000))
        # Independently fixed synthetic physical-terminal example: RA=0,
        # LA=2, LL=6. After RA/LL swap, the limb leads are below.
        expected = np.asarray([-4, -6, -2, 5, -1, -4, 16, 17, 18, 19, 20, 21], dtype=np.float32)
        np.testing.assert_array_equal(observed[0, :, 0], expected)
        self.assertEqual(float(observed[0, 3, 0]), 5.0)  # aVR
        self.assertEqual(float(observed[0, 5, 0]), -4.0)  # aVF
        independent = np.concatenate((x[:, :2], x[:, 6:]), axis=1)
        library_result = derive_display(named_reversal(independent, "RA_LL_reversal")).astype(np.float32)
        np.testing.assert_array_equal(observed, library_result)

    def test_all_conditions_use_12_by_1000_and_frozen_mask_sentinel(self):
        raw = np.zeros((1, 12, 5000), dtype=np.float32)
        raw[:, 0] = 2
        raw[:, 1] = 6
        mean = np.arange(12, dtype=np.float32)
        std = np.ones(12, dtype=np.float32) * 2
        arrays = normalized_conditions(raw, mean, std, ((0, 3, 5, 8, 11),))
        self.assertEqual(len(arrays), 3)
        self.assertTrue(all(array.shape == (1, 12, 1000) for array in arrays))
        np.testing.assert_array_equal(arrays[2][0, [0, 3, 5, 8, 11], :], 0)
        expected_reverse = (exact_ra_ll(to_ningbo_time(raw)) - mean[None, :, None]) / std[None, :, None]
        np.testing.assert_array_equal(arrays[1], expected_reverse)


if __name__ == "__main__":
    unittest.main()
