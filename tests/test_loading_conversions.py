import unittest

import pandas as pd

from pipeline.loading_conversion_normalizer import normalize_entity_loading_dataframe


class LoadingConversionTests(unittest.TestCase):
    def convert(self, rows):
        dataframe = pd.DataFrame(rows)
        converted, changed = normalize_entity_loading_dataframe(dataframe)
        return converted, changed

    def test_phr_to_total_formulation_wt(self):
        converted, changed = self.convert(
            [
                {
                    "article_index": 1,
                    "sample_index_in_article": 1,
                    "sample_name": "S1",
                    "entity_role": "matrix_polymer",
                    "entity_loading": None,
                    "entity_loading_unit": "#",
                    "entity_loading_basis": "#",
                },
                {
                    "article_index": 1,
                    "sample_index_in_article": 1,
                    "sample_name": "S1",
                    "entity_role": "main_flame_retardant",
                    "entity_loading": 30,
                    "entity_loading_unit": "phr",
                    "entity_loading_basis": "phr_resin",
                },
            ]
        )
        self.assertAlmostEqual(converted.at[1, "entity_loading_phr_resin"], 30.0)
        self.assertAlmostEqual(
            converted.at[1, "entity_loading_wt_total_formulation"],
            30.0 / 130.0 * 100.0,
            places=6,
        )
        self.assertGreater(sum(changed.values()), 0)

    def test_total_formulation_wt_to_phr(self):
        converted, _ = self.convert(
            [
                {
                    "sample_uid": "sample-1",
                    "entity_role": "matrix_polymer",
                    "entity_loading": 70,
                    "entity_loading_unit": "wt%",
                    "entity_loading_basis": "wt_total",
                },
                {
                    "sample_uid": "sample-1",
                    "entity_role": "main_flame_retardant",
                    "entity_loading": 30,
                    "entity_loading_unit": "wt%",
                    "entity_loading_basis": "wt_total",
                },
            ]
        )
        self.assertAlmostEqual(converted.at[1, "entity_loading_wt_total_formulation"], 30.0)
        self.assertAlmostEqual(converted.at[1, "entity_loading_phr_resin"], 30.0 / 70.0 * 100.0, places=6)

    def test_parts_by_weight_to_both_normalized_fields(self):
        converted, _ = self.convert(
            [
                {
                    "sample_uid": "sample-2",
                    "entity_role": "matrix_polymer",
                    "entity_loading": 100,
                    "entity_loading_unit": "parts",
                    "entity_loading_basis": "parts_by_weight",
                },
                {
                    "sample_uid": "sample-2",
                    "entity_role": "main_flame_retardant",
                    "entity_loading": 30,
                    "entity_loading_unit": "parts",
                    "entity_loading_basis": "parts_by_weight",
                },
            ]
        )
        self.assertAlmostEqual(
            converted.at[1, "entity_loading_wt_total_formulation"],
            30.0 / 130.0 * 100.0,
            places=6,
        )
        self.assertAlmostEqual(converted.at[1, "entity_loading_phr_resin"], 30.0)

    def test_incomplete_mixed_basis_is_not_cross_converted(self):
        converted, _ = self.convert(
            [
                {
                    "sample_uid": "sample-3",
                    "entity_role": "matrix_polymer",
                    "entity_loading": None,
                    "entity_loading_unit": "#",
                    "entity_loading_basis": "#",
                },
                {
                    "sample_uid": "sample-3",
                    "entity_role": "main_flame_retardant",
                    "entity_loading": 30,
                    "entity_loading_unit": "phr",
                    "entity_loading_basis": "phr_resin",
                },
                {
                    "sample_uid": "sample-3",
                    "entity_role": "synergistic_flame_retardant",
                    "entity_loading": 5,
                    "entity_loading_unit": "wt%",
                    "entity_loading_basis": "wt_total",
                },
            ]
        )
        self.assertTrue(pd.isna(converted.at[1, "entity_loading_wt_total_formulation"]))
        self.assertTrue(pd.isna(converted.at[2, "entity_loading_phr_resin"]))

    def test_existing_normalized_values_are_preserved(self):
        converted, changed = self.convert(
            [
                {
                    "sample_uid": "sample-4",
                    "entity_role": "matrix_polymer",
                    "entity_loading": 100,
                    "entity_loading_unit": "parts",
                    "entity_loading_basis": "parts_by_weight",
                    "entity_loading_wt_total_formulation": 99.123,
                },
                {
                    "sample_uid": "sample-4",
                    "entity_role": "main_flame_retardant",
                    "entity_loading": 30,
                    "entity_loading_unit": "parts",
                    "entity_loading_basis": "parts_by_weight",
                    "entity_loading_phr_resin": 29.999,
                },
            ]
        )
        self.assertEqual(converted.at[0, "entity_loading_wt_total_formulation"], 99.123)
        self.assertEqual(converted.at[1, "entity_loading_phr_resin"], 29.999)
        self.assertAlmostEqual(converted.at[1, "entity_loading_wt_total_formulation"], 30 / 130 * 100, places=6)
        self.assertAlmostEqual(converted.at[0, "entity_loading_phr_resin"], 100.0)
        self.assertEqual(changed["entity_loading_wt_total_formulation"], 1)
        self.assertEqual(changed["entity_loading_phr_resin"], 1)


if __name__ == "__main__":
    unittest.main()
