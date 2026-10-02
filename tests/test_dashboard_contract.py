import json
import unittest
from pathlib import Path


class DashboardContractTests(unittest.TestCase):
    def test_next_dashboard_declares_required_v1_pages(self):
        required = ["app/page.tsx", "app/scanner/page.tsx", "app/portfolio/page.tsx", "app/positions/page.tsx", "app/models/page.tsx", "app/laya/page.tsx", "app/research/page.tsx", "app/backtests/page.tsx", "app/execution/page.tsx", "app/replay/page.tsx", "app/risk/page.tsx", "app/system/page.tsx"]
        for relative in required:
            self.assertTrue((Path("dashboard") / relative).is_file(), relative)

    def test_dashboard_has_no_live_capital_control_surface(self):
        text = "\n".join(path.read_text(encoding="utf-8") for path in Path("dashboard").rglob("*.tsx"))
        self.assertIn("LIVE CAPITAL LOCKED", text)
        self.assertNotIn("tiny-live", text.lower())

    def test_package_is_next_typescript(self):
        package = json.loads(Path("dashboard/package.json").read_text(encoding="utf-8"))
        self.assertIn("next", package["dependencies"])
        self.assertIn("typescript", package["devDependencies"])


if __name__ == "__main__":
    unittest.main()
