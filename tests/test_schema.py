from __future__ import annotations

import unittest

import pandas as pd

from pipeline.schema import canonicalize_dataframe_columns


class SchemaCompatibilityTests(unittest.TestCase):
    def test_canonical_headers_are_preserved(self):
        frame = pd.DataFrame(
            [{"article_index": 7, "sample_name": "S1", "DOI": "10.1/example"}]
        )

        canonical = canonicalize_dataframe_columns(frame, context="canonical fixture")

        self.assertEqual(
            list(canonical.columns),
            ["article_index", "sample_name", "DOI"],
        )
        self.assertEqual(canonical.loc[0, "article_index"], 7)
        self.assertEqual(canonical.loc[0, "sample_name"], "S1")

    def test_unknown_english_headers_are_preserved(self):
        frame = pd.DataFrame([{"custom_review_note": "keep"}])
        canonical = canonicalize_dataframe_columns(frame, context="custom fixture")
        self.assertEqual(list(canonical.columns), ["custom_review_note"])
        self.assertEqual(canonical.loc[0, "custom_review_note"], "keep")


if __name__ == "__main__":
    unittest.main()
