from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

from pipeline import eva_split_pipeline as pipeline


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "benchmark" / "gt50" / "generation" / "eva_gt50_ablation_runner.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("gt50_ablation_generation", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GT50GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner()

    def test_only_paper_ablation_configurations_are_exposed(self) -> None:
        self.assertEqual(
            self.runner.PAPER_CONFIGURATIONS,
            (
                "direct_naive_tri_view",
                "expert_tri_view_single_shot",
                "split_naive",
                "full_pipeline",
            ),
        )

    def test_frozen_prompt_hashes_are_stable(self) -> None:
        self.runner.validate_prompt_snapshots()
        self.assertEqual(
            self.runner.prompt_sha256(self.runner.FROZEN_ENTITY_FORMULATION_PROMPT),
            "b87eaf014cbfff8b2c012422087da0185507a2e73e4a3601ddb65cf63bdb7f9d",
        )
        self.assertEqual(
            self.runner.prompt_sha256(self.runner.FROZEN_PERFORMANCE_PROMPT),
            "1e38873154896834c578e58b17183b2bd38a4b13e653f3934f853f496ad23c33",
        )

    def test_visual_temperature_is_zero_and_text_stages_remain_point_one(self) -> None:
        class StubResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "{}"}}]}

        class RecordingSession:
            def __init__(self) -> None:
                self.payloads = []

            def post(self, *args, **kwargs):
                self.payloads.append(kwargs["json"])
                return StubResponse()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formulation = pipeline.EntityFormulationAnalyzer(
                output_excel=str(root / "formulation" / "formulation_entities.xlsx"),
                parts_excel=str(root / "formulation" / "formulation_entity_parts.xlsx"),
                model_name="test-model",
                input_folder=str(root),
            )
            performance = pipeline.PerformanceAnalyzer(
                output_excel=str(root / "performance" / "performance_samples.xlsx"),
                model_name="test-model",
                input_folder=str(root),
            )
            visual = pipeline.VisualPerformanceAnalyzer(
                output_excel=str(root / "visual" / "performance_samples_visual.xlsx"),
                model_name="test-model",
                input_folder=str(root),
            )
            self.assertEqual(formulation.temperature, 0.1)
            self.assertEqual(performance.temperature, 0.1)
            self.assertEqual(visual.temperature, 0.0)
            formulation.session = RecordingSession()
            visual.session = RecordingSession()
            formulation.call_llm_api([], file_id="formulation-temperature-test")
            visual.call_llm_api([], file_id="visual-temperature-test")
            self.assertEqual(formulation.session.payloads[0]["temperature"], 0.1)
            self.assertEqual(visual.session.payloads[0]["temperature"], 0.0)

    def test_one_shot_postprocessing_maps_frozen_schema_to_public_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analyzer = self.runner.TriViewSingleShotAnalyzer(
                output_excel=str(root / "formulation_entities.xlsx"),
                parts_excel=str(root / "formulation_entity_parts.xlsx"),
                performance_excel=str(root / "performance_samples_final.xlsx"),
                model_name="test-model",
                input_folder=str(root),
                prompt_template=self.runner.DIRECT_NAIVE_TRI_VIEW_PROMPT,
            )
            response = {
                "article_title": "Test paper",
                "DOI": "10.0000/test",
                "formulation_entities": [
                    {
                        "article_sample_order": 1,
                        "sample_name": "S1",
                        "entity_name": "APP",
                        "entity_loading_text": "20 wt%",
                        "entity_loading_wt_total": 20,
                        "entity_loading_conversion_note": "reported",
                        "entity_trade_name": "Grade A",
                    }
                ],
                "entity_parts": [
                    {
                        "article_sample_order": 1,
                        "sample_name": "S1",
                        "entity_name": "APP",
                        "part_name": "core",
                        "relation_to_entity": "core_of",
                    }
                ],
                "performance_samples": [
                    {
                        "article_sample_order": 1,
                        "sample_name": "S1",
                        "TGAresidue": 12,
                        "cone_sample_thickness": 3,
                        "residue_rate": 18,
                    }
                ],
            }
            entity_rows = analyzer.parse_task_response(json.dumps(response), article_idx=7)
            self.assertEqual(entity_rows[0]["entity_loading_raw"], "20 wt%")
            self.assertEqual(entity_rows[0]["entity_loading_wt_total_formulation"], 20)
            self.assertEqual(entity_rows[0]["entity_loading_conversion_notes"], "reported")
            self.assertEqual(entity_rows[0]["entity_grade"], "Grade A")
            self.assertEqual(analyzer._pending_parts()[0]["part_relation"], "core_of")
            performance = analyzer._pending_performance()[0]
            self.assertEqual(performance["TGA_residue"], 12)
            self.assertEqual(performance["cone_specimen_thickness"], 3)
            self.assertEqual(performance["cone_residue"], 18)

    def test_public_runner_has_no_private_paths_han_emoji_or_legacy_variants(self) -> None:
        text = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"[\u3400-\u9fff\uf900-\ufaff]", text))
        self.assertIsNone(re.search(r"[\U0001f300-\U0001faff]", text))
        self.assertIsNone(re.search(r"(?i)[A-Z]:\\(?:Users|ai_data|home|tmp)\\", text))
        self.assertNotIn('"direct_naive"', text)
        self.assertNotIn('"expert_single_shot"', text)


if __name__ == "__main__":
    unittest.main()
