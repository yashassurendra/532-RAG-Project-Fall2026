import unittest

import numpy as np

from verify_and_test_pipeline import (
    compute_exact_ground_truth,
    format_suffix,
    generate_selectivity_masks,
)


class GroundTruthTests(unittest.TestCase):
    def test_expected_suffixes(self):
        self.assertEqual(format_suffix(1_000), "test1k")
        self.assertEqual(format_suffix(100_000), "100k")
        self.assertEqual(format_suffix(171_332), "171332")

    def test_masks_are_exact_and_nested(self):
        masks = generate_selectivity_masks(1_000, seed=42)
        self.assertEqual(int(masks["1pct"].sum()), 10)
        self.assertEqual(int(masks["10pct"].sum()), 100)
        self.assertEqual(int(masks["50pct"].sum()), 500)
        self.assertTrue(np.all(masks["1pct"] <= masks["10pct"]))
        self.assertTrue(np.all(masks["10pct"] <= masks["50pct"]))

    def test_exact_ground_truth_obeys_filter(self):
        passages = np.eye(4, dtype=np.float32)
        queries = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        masks = {
            "unfiltered": np.ones(4, dtype=bool),
            "subset": np.array([True, False, True, False]),
        }
        result = compute_exact_ground_truth(queries, passages, masks, k=1)
        self.assertEqual(result["unfiltered"].tolist(), [[1]])
        self.assertIn(result["subset"][0, 0], (0, 2))
        self.assertTrue(masks["subset"][result["subset"]].all())


if __name__ == "__main__":
    unittest.main()
