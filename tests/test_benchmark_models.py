import unittest

from research.benchmark_models import available_candidates


class BenchmarkModelTests(unittest.TestCase):
    def test_core_candidates_are_always_available(self):
        candidates = available_candidates()
        self.assertIn("xgboost", candidates)
        self.assertIn("hist_gradient_boosting", candidates)


if __name__ == "__main__":
    unittest.main()
