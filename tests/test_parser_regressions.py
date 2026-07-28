from __future__ import annotations

import json
import unittest

import pandas as pd

from pipeline import LLMvision_extractor as vision
from pipeline import clean_dict
from pipeline import eva_split_pipeline as pipeline


class ParserRegexRegressionTests(unittest.TestCase):
    def test_json_fences_and_numeric_normalization(self):
        analyzer = pipeline.EvaPipelineBaseAnalyzer.__new__(pipeline.EvaPipelineBaseAnalyzer)

        self.assertEqual(
            json.loads(analyzer._smart_fix_json("```json\n{\"value\": 1}\n```")),
            {"value": 1},
        )
        self.assertEqual(
            json.loads(analyzer._smart_fix_json("```\n{\"value\": 1}\n```")),
            {"value": 1},
        )
        self.assertEqual(pipeline.normalize_exact_dedupe_value("w-1.00e+2"), "w-100")
        self.assertEqual(pipeline.numeric_value("value=-12.5 wt%"), -12.5)

    def test_material_and_metric_patterns(self):
        self.assertEqual(
            pipeline.detect_matrix_components("EVA/LLDPE/ethylene-octene"),
            (True, True, True, False),
        )
        self.assertEqual(pipeline.normalize_metric_output_value("w-25.0 kW/m2"), "w-25.0")
        self.assertEqual(clean_dict.split_name_tokens("APP; MPP, CNT"), ["APP", "MPP", "CNT"])
        self.assertIsNotNone(
            clean_dict.find_special_material_rule("20 wt% CNT", allow_variant=True)
        )
        eg_details = clean_dict.match_name_details("EG2")
        self.assertEqual(eg_details["canonical_name"], "EG")
        self.assertEqual(eg_details["smiles"], "[C].[O-]S(=O)(=O)[O-]")
        self.assertIn("Representative SMILES Proxy", eg_details["source"])
        self.assertFalse(eg_details["is_molecular"])
        self.assertIsNone(
            clean_dict.match_name_details("silicate-assisted expandable graphite")
        )

    def test_visual_target_patterns(self):
        for expression in (clean_dict.VISUAL_POSITIVE_RE, vision.VISUAL_POSITIVE_RE):
            self.assertIsNotNone(expression.search("new flame retardant repeating unit"))
        for expression in (clean_dict.HARD_VISUAL_SKIP_RE, vision.HARD_VISUAL_SKIP_RE):
            self.assertIsNotNone(expression.search("core shell morphology"))
        for expression in (clean_dict.COMMON_NON_MOLECULAR_RE, vision.NON_MOLECULAR_NAME_RE):
            self.assertIsNotNone(expression.search("CNT composite"))
        self.assertEqual(
            vision.IMAGE_ID_RE.search("ID: Figure-1.jpg").group(1),
            "Figure-1.jpg",
        )


class PaperMergeRegressionTests(unittest.TestCase):
    @staticmethod
    def frame(sample_order, loi, signature, record_uid):
        return pd.DataFrame(
            [
                {
                    "article_index": 7,
                    "sample_index_in_article": sample_order,
                    "sample_name": "EVA-1",
                    "performance_condition_signature": signature,
                    "performance_record_uid": record_uid,
                    "LOI": loi,
                    "performance_notes": None,
                }
            ]
        )

    def test_same_article_order_and_name_merge_across_conditions(self):
        primary = self.frame(1, None, "cone_heat_flux=35", "record-text")
        visual = self.frame(1, "w-28.5%", "cone_heat_flux=50", "record-visual")

        merged, filled = pipeline.merge_text_and_visual_frames(primary, visual)

        self.assertEqual(len(merged), 1)
        self.assertEqual(filled, 1)
        self.assertEqual(merged.at[0, "LOI"], "w-28.5")

    def test_conflicting_shared_conditions_stay_visual_only(self):
        primary = self.frame(1, None, "cone_specimen_thickness=3", "record-text")
        visual = self.frame(1, "w-30%", "cone_specimen_thickness=2.94", "record-visual")
        primary["cone_specimen_thickness"] = [3]
        visual["cone_specimen_thickness"] = ["w-2.94"]

        merged, filled = pipeline.merge_text_and_visual_frames(primary, visual)

        self.assertEqual(len(merged), 2)
        self.assertEqual(filled, 0)

    def test_different_sample_order_stays_visual_only(self):
        primary = self.frame(1, None, "default", "record-text")
        visual = self.frame(2, "w-30%", "default", "record-visual")

        merged, filled = pipeline.merge_text_and_visual_frames(primary, visual)

        self.assertEqual(len(merged), 2)
        self.assertEqual(filled, 0)
        self.assertTrue(pd.isna(merged.at[0, "LOI"]))
        self.assertEqual(merged.iloc[1]["LOI"], "w-30")
        self.assertIn("supplement-only row", merged.iloc[1]["performance_notes"])


if __name__ == "__main__":
    unittest.main()
