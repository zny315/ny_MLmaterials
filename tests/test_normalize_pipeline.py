from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"


def load_normalizer():
    if str(PIPELINE_DIR) not in sys.path:
        sys.path.insert(0, str(PIPELINE_DIR))
    path = PIPELINE_DIR / "normalize_pipeline.py"
    spec = importlib.util.spec_from_file_location("normalize_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_entities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "article_index": 1,
                "sample_uid": "sample-1",
                "entity_role": "matrix_polymer",
                "entity_loading": None,
                "entity_loading_unit": "#",
                "entity_loading_basis": "#",
                "entity_loading_wt_total_formulation": 62.5,
            },
            {
                "article_index": 1,
                "sample_uid": "sample-1",
                "entity_role": "main_flame_retardant",
                "entity_loading": 60.0,
                "entity_loading_unit": "phr",
                "entity_loading_basis": "phr_resin",
            },
        ]
    )


class NormalizePipelineTests(unittest.TestCase):
    def test_deterministic_stages_apply_fill_only_loading_conversion(self):
        module = load_normalizer()
        entities_path = Path("output") / "formulation_entities.xlsx"
        writes = []
        with (
            patch.object(module, "fix_excel", return_value={"changed_cells": 0}),
            patch.object(module, "read_table", return_value=make_entities()),
            patch.object(module, "write_table", side_effect=lambda frame, path: writes.append((frame, path))),
        ):
            module.run_deterministic_stages(
                entities_path.parent,
                [entities_path],
                skip_dedupe=True,
                skip_dictionary=True,
            )

        normalized, written_path = writes[-1]
        self.assertEqual(written_path, entities_path)
        self.assertEqual(normalized.at[0, "entity_loading_wt_total_formulation"], 62.5)
        self.assertEqual(normalized.at[1, "entity_loading_phr_resin"], 60.0)
        self.assertEqual(normalized.at[1, "entity_loading_wt_total_formulation"], 37.5)


if __name__ == "__main__":
    unittest.main()
