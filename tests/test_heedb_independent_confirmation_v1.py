import unittest

import numpy as np

from src import heedb_independent_confirmation_v1 as confirmation
from src import heedb_external_corrected_v2 as corrected
from src.ecg_physics import derive_display, named_reversal


class ConfirmationInputTests(unittest.TestCase):
    def test_exact_ningbo_shape_and_sampling(self):
        raw = np.broadcast_to(np.arange(5000, dtype=np.float32), (2, 12, 5000)).copy()
        clean, reverse = confirmation.model_inputs(raw, np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.float32))
        self.assertEqual(clean.shape, (2, 12, 1000))
        self.assertEqual(reverse.shape, (2, 12, 1000))
        np.testing.assert_array_equal(clean[0, 0], np.arange(0, 5000, 5, dtype=np.float32))
        with self.assertRaises(ValueError):
            confirmation.model_inputs(raw[:, :, :-1], np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.float32))

    def test_all_twelve_reversed_leads_match_physics_library(self):
        raw = np.zeros((1, 12, 5000), dtype=np.float32)
        raw[:, 0] = 2  # I
        raw[:, 1] = 6  # II
        for lead in range(6, 12):
            raw[:, lead] = 10 + lead
        sampled = corrected.to_ningbo_time(raw)
        independent = np.concatenate((sampled[:, :2], sampled[:, 6:]), axis=1)
        expected = derive_display(named_reversal(independent, "RA_LL_reversal")).astype(np.float32)
        actual = corrected.exact_ra_ll(sampled)
        self.assertEqual(actual.shape, (1, 12, 1000))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[0, :, 0], np.array([-4, -6, -2, 5, -1, -4, 16, 17, 18, 19, 20, 21], dtype=np.float32))
        self.assertEqual(float(actual[0, 3, 0]), 5.0)  # aVR
        self.assertEqual(float(actual[0, 5, 0]), -4.0)  # aVF

    def test_normalization_occurs_after_reversal(self):
        raw = np.zeros((1, 12, 5000), dtype=np.float32)
        raw[:, 0] = 2
        raw[:, 1] = 6
        mean = np.arange(12, dtype=np.float32)
        std = np.arange(1, 13, dtype=np.float32)
        clean, reverse = confirmation.model_inputs(raw, mean, std)
        sampled = corrected.to_ningbo_time(raw)
        expected = (corrected.exact_ra_ll(sampled) - mean[None, :, None]) / std[None, :, None]
        np.testing.assert_allclose(reverse, expected.astype(np.float32), rtol=0, atol=0)
        self.assertEqual(clean.shape[1:], (12, 1000))


if __name__ == "__main__":
    unittest.main()
