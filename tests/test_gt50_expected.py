from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GT50ExpectedTests(unittest.TestCase):
    def test_expected_metrics_cover_all_rows(self):
        payload = json.loads((ROOT / "benchmark" / "gt50" / "expected_metrics.json").read_text(encoding="utf-8"))
        rows = payload["metrics"]
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            {(row["method"], row["view"]) for row in rows},
            {
                (method, view)
                for method in ("Direct LLM", "Expert Prompt Only", "Task Decomposition Only", "Full Pipeline")
                for view in ("entities", "parts", "performance", "all")
            },
        )

    def test_full_pipeline_expected_f1_matches_paper(self):
        payload = json.loads((ROOT / "benchmark" / "gt50" / "expected_metrics.json").read_text(encoding="utf-8"))
        rows = {(row["method"], row["view"]): row for row in payload["metrics"]}
        self.assertEqual(rows[("Full Pipeline", "entities")]["F1"], 0.9175698916130064)
        self.assertEqual(rows[("Full Pipeline", "parts")]["F1"], 0.7574338085539716)
        self.assertEqual(rows[("Full Pipeline", "performance")]["F1"], 0.9617958728388176)
        self.assertEqual(rows[("Full Pipeline", "all")]["F1"], 0.8876771044171924)


if __name__ == "__main__":
    unittest.main()
