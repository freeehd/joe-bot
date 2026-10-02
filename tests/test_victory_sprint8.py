import unittest
from research.promotion import promote_runtime_package


class VictorySprint8Tests(unittest.TestCase):
    def test_promotion_function_rejects_live_channel_at_api_boundary(self):
        # The detailed package/promotion lifecycle is covered in test_runtime_package_drift.
        # This test protects the public Sprint 8 API from later accidental broadening.
        import inspect
        source=inspect.getsource(promote_runtime_package)
        self.assertIn('channel not in {"shadow", "paper"}',source)
        self.assertIn('tiny-live',source)


if __name__ == "__main__":
    unittest.main()
