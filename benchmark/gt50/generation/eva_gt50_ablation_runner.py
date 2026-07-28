#!/usr/bin/env python3
"""Run the four GT50 extraction ablations reported in the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_DIR = REPO_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import eva_split_pipeline as eva


PAPER_CONFIGURATIONS = (
    "direct_naive_tri_view",
    "expert_tri_view_single_shot",
    "split_naive",
    "full_pipeline",
)
DEFAULT_INPUT = REPO_ROOT / "inputs" / "mineru"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "gt50_generation"
DEFAULT_GT_WORKBOOK = (
    REPO_ROOT
    / "benchmark"
    / "gt50"
    / "ground_truth"
    / "GT50_EVA_hierarchical_ground_truth_publication_clean.xlsx"
)
DEFAULT_INPUT_ORDER = REPO_ROOT / "data" / "full_corpus_input_order.csv"

PROMPT_SNAPSHOT_DIR = Path(__file__).with_name("prompts")


def load_prompt_snapshot(filename: str) -> str:
    return (PROMPT_SNAPSHOT_DIR / filename).read_text(encoding="utf-8")


TRI_VIEW_JSON_SCHEMA = load_prompt_snapshot("tri_view_schema.txt")
EXPERT_TRI_VIEW_JSON_SCHEMA = load_prompt_snapshot("expert_tri_view_schema.txt")
DIRECT_NAIVE_TRI_VIEW_PROMPT = load_prompt_snapshot("v0_direct_naive_tri_view.txt")
SPLIT_NAIVE_FORMULATION_PROMPT = load_prompt_snapshot("v2_split_naive_formulation.txt")
SPLIT_NAIVE_PERFORMANCE_PROMPT = load_prompt_snapshot("v2_split_naive_performance.txt")

MULTI_CONDITION_PARAGRAPH = (
    "If the same formulation sample is tested under multiple conditions, emit one separate "
    "samples-array record per condition. Repeat the exact sample_name, keep every "
    "condition-specific record, and state the distinguishing condition in performance_notes. "
    "Never average, overwrite, or deduplicate records merely because sample_name is the same.\n\n"
)
FROZEN_ENTITY_FORMULATION_PROMPT = eva.ENTITY_FORMULATION_PROMPT
FROZEN_PERFORMANCE_PROMPT = eva.PERFORMANCE_PROMPT.replace(MULTI_CONDITION_PARAGRAPH, "")

EXPERT_TRI_VIEW_SINGLE_SHOT_PROMPT = f"""
You are a materials researcher building a standardized EVA/PE/POE flame-retardant database.

This is a ONE-SHOT extraction task. Do NOT split the work into separate requests.
Extract complete sample-level formulation entities, entity parts, and performance data from the supplied paper content.
Use the same expert rules as the full pipeline, but return one JSON object with the three target database views:
formulation_entities, entity_parts, and performance_samples.

The detailed formulation rules from the pipeline are included first, followed by the detailed performance rules.
Where the original pipeline says FORMULATION ONLY or PERFORMANCE ONLY, interpret those rules as applying to the relevant view in this single combined output.

{FROZEN_ENTITY_FORMULATION_PROMPT.replace("Paper Content:", "")}

{FROZEN_PERFORMANCE_PROMPT.replace("Paper Content:", "")}

{EXPERT_TRI_VIEW_JSON_SCHEMA}

Paper Content:
"""

PROMPT_SHA256 = {
    "tri_view_schema": "28253b0e44568b8589511a03cc95c8a7d0de9da893ccafd1b3cf6efa45f222b0",
    "expert_tri_view_schema": "eb48a72f5899b065467d39475fabad7111af0a1a83ddec78ba783a9aa41bcb54",
    "direct_naive_tri_view": "f5426ae3ba099f8e7747150ee882b0387b86ac75ee5ce1fa23f43a132d46505f",
    "expert_tri_view_single_shot": "758268157b57b83a8500da04447e3c052e5cc72f49c51f846ca98f2e13ed7873",
    "split_naive_formulation": "0f8a2ee4fc1c399f0df618726aa57cee25808830c79955bd13abb4c6cf457f19",
    "split_naive_performance": "467764ad0db0e437c984f797a77024ee02aa414b421290507a52a961229ee0c6",
    "full_pipeline_formulation": "b87eaf014cbfff8b2c012422087da0185507a2e73e4a3601ddb65cf63bdb7f9d",
    "full_pipeline_performance": "1e38873154896834c578e58b17183b2bd38a4b13e653f3934f853f496ad23c33",
}


def prompt_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_prompt_snapshots() -> None:
    snapshots = {
        "tri_view_schema": TRI_VIEW_JSON_SCHEMA,
        "expert_tri_view_schema": EXPERT_TRI_VIEW_JSON_SCHEMA,
        "direct_naive_tri_view": DIRECT_NAIVE_TRI_VIEW_PROMPT,
        "expert_tri_view_single_shot": EXPERT_TRI_VIEW_SINGLE_SHOT_PROMPT,
        "split_naive_formulation": SPLIT_NAIVE_FORMULATION_PROMPT,
        "split_naive_performance": SPLIT_NAIVE_PERFORMANCE_PROMPT,
        "full_pipeline_formulation": FROZEN_ENTITY_FORMULATION_PROMPT,
        "full_pipeline_performance": FROZEN_PERFORMANCE_PROMPT,
    }
    mismatches = {
        name: prompt_sha256(value)
        for name, value in snapshots.items()
        if prompt_sha256(value) != PROMPT_SHA256[name]
    }
    if mismatches:
        details = ", ".join(f"{name}={digest}" for name, digest in sorted(mismatches.items()))
        raise RuntimeError(f"Frozen GT50 prompt snapshot mismatch: {details}")


validate_prompt_snapshots()


@dataclass(frozen=True)
class ArticleInput:
    article_index: int
    source_id: str
    paper_dir: Path


def copy_args(args: argparse.Namespace, **updates) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "model"


def entity_output_columns() -> list[str]:
    columns = list(eva.ENTITY_COLUMNS)
    for column in ("entity_smiles", "entity_smiles_source"):
        if column not in columns:
            columns.append(column)
    return columns


ENTITY_OUTPUT_COLUMNS = entity_output_columns()

ENTITY_JSON_TO_COLUMN = {
    "matrix_system": "matrix_system",
    "formulation_measurement_basis": "formulation_basis",
    "EVA_fraction_in_matrix": "EVA_polymer_fraction",
    "PE_fraction_in_matrix": "PE_polymer_fraction",
    "POE_fraction_in_matrix": "POE_polymer_fraction",
    "EVA_VA_content": "EVA_VA_content",
    "PE_type": "PE_type",
    "POE_type": "POE_type",
    "matrix": "matrix",
    "matrix_key_data": "key_content",
    "melt_flow_index": "melt_index",
    "matrix_grade_information": "matrix_trname",
    "total_additive_loading": "total_fr_loading",
    "mixing_method": "mixing_method",
    "formulation_notes": "formulation_notes",
    "entity_id": "entity_id",
    "entity_name": "entity_name",
    "entity_role": "entity_role",
    "entity_type": "entity_type",
    "entity_loading": "entity_loading",
    "entity_loading_raw": "entity_loading_text",
    "entity_loading_unit": "entity_loading_unit",
    "entity_loading_basis": "entity_loading_basis",
    "entity_loading_wt_total_formulation": "entity_loading_wt_total",
    "entity_loading_phr_resin": "entity_loading_phr_resin",
    "entity_loading_conversion_notes": "entity_loading_conversion_note",
    "entity_material_type": "entity_material_type",
    "entity_grade": "entity_trade_name",
    "entity_profile": "entity_profile",
    "entity_figure_locator": "entity_fig_locator",
    "entity_bbox": "entity_bbox",
    "entity_smiles": "entity_smiles",
    "entity_smiles_source": "entity_smiles_source",
}

PART_JSON_TO_COLUMN = {
    "entity_id": "entity_id",
    "entity_name": "entity_name",
    "part_name": "part_name",
    "part_role": "part_role",
    "part_relation": "relation_to_entity",
    "modification_method": "modification_method",
    "part_ratio": "part_ratio",
    "part_loading": "part_loading",
    "part_loading_unit": "part_loading_unit",
    "part_figure_locator": "part_fig_locator",
    "part_bbox": "part_bbox",
    "part_smiles": "part_smiles",
    "part_smiles_source": "part_smiles_source",
    "part_notes": "part_notes",
}

PERFORMANCE_JSON_TO_COLUMN = {
    "LOI": "LOI",
    "UL94_rating": "UL94_rating",
    "elongation_at_break": "elongation_at_break",
    "tensile_strength": "tensile_strength",
    "T1": "T1",
    "T5": "T5",
    "T10": "T10",
    "T50": "T50",
    "Tmax1": "Tmax1",
    "Tmax2": "Tmax2",
    "TGA_residue": "TGAresidue",
    "TTI": "TTI",
    "cone_heat_flux": "cone_heat_flux",
    "cone_specimen_thickness": "cone_sample_thickness",
    "pHRR": "pHRR",
    "THR": "THR",
    "pSPR": "pSPR",
    "TSP": "TSP",
    "pCOR": "pCOR",
    "pCO2R": "pCO2R",
    "cone_residue": "residue_rate",
    "performance_notes": "performance_notes",
}


def load_gt50_articles(
    input_root: Path,
    gt_workbook: Path,
    input_order_csv: Path,
) -> list[ArticleInput]:
    selected = pd.read_excel(gt_workbook, sheet_name="selected_articles")
    input_order = pd.read_csv(input_order_csv)
    required_selected = {"article_index"}
    required_order = {"article_index", "source_id"}
    if not required_selected.issubset(selected.columns):
        raise ValueError("GT50 selected_articles sheet is missing article_index.")
    if not required_order.issubset(input_order.columns):
        raise ValueError("Input-order CSV must contain article_index and source_id.")

    selected = selected[["article_index"]].copy()
    selected["article_index"] = pd.to_numeric(selected["article_index"], errors="raise").astype(int)
    input_order = input_order[["article_index", "source_id"]].copy()
    input_order["article_index"] = pd.to_numeric(input_order["article_index"], errors="raise").astype(int)
    merged = selected.merge(input_order, on="article_index", how="left", validate="one_to_one")
    if merged["source_id"].isna().any():
        missing = merged.loc[merged["source_id"].isna(), "article_index"].tolist()
        raise ValueError(f"GT50 article indexes are missing from the input-order CSV: {missing}")
    if len(merged) != 50:
        raise ValueError(f"Expected 50 GT50 articles, found {len(merged)}.")

    articles = []
    for row in merged.itertuples(index=False):
        source_id = str(row.source_id).strip()
        paper_dir = (input_root / source_id).resolve()
        if not paper_dir.is_dir():
            raise FileNotFoundError(f"MinerU paper directory not found: {paper_dir}")
        articles.append(
            ArticleInput(
                article_index=int(row.article_index),
                source_id=source_id,
                paper_dir=paper_dir,
            )
        )
    return articles


def configure_analyzer(analyzer, args: argparse.Namespace) -> None:
    keys = eva.load_api_keys(args.api_keys_file) if args.api_keys_file else []
    if args.api_key_limit:
        keys = keys[: args.api_key_limit]
    eva.configure_runtime(analyzer, args, eva.make_api_key_queue(keys))


def process_articles(
    analyzer,
    articles: list[ArticleInput],
    workers: int,
) -> None:
    selected = articles[: analyzer.limit] if analyzer.limit and analyzer.limit > 0 else articles
    if workers <= 1:
        for article in selected:
            if not analyzer.running:
                break
            analyzer.process_paper_dir(str(article.paper_dir), article_idx=article.article_index)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                analyzer.process_paper_dir,
                str(article.paper_dir),
                article.article_index,
            ): article
            for article in selected
        }
        for future in as_completed(futures):
            article = futures[future]
            if not analyzer.running:
                break
            try:
                future.result()
            except Exception as exc:
                analyzer.run_failed += 1
                analyzer.log(
                    f"[{analyzer.task_name}] concurrent task failed for "
                    f"{article.source_id}: {exc}"
                )


class TriViewSingleShotAnalyzer(eva.SplitTaskAnalyzer):
    task_name = "expert_tri_view_single_shot"
    include_images = True

    def __init__(
        self,
        output_excel: str,
        parts_excel: str,
        performance_excel: str,
        model_name: str,
        input_folder: str,
        prompt_template: str,
    ):
        super().__init__(
            output_excel=output_excel,
            model_name=model_name,
            input_folder=input_folder,
        )
        self.parts_excel = parts_excel
        self.performance_excel = performance_excel
        self.prompt_template = prompt_template
        self.column_order = ENTITY_OUTPUT_COLUMNS
        self._pending_context = threading.local()

    def _set_pending(self, parts: list[dict], performance: list[dict]) -> None:
        self._pending_context.parts = parts
        self._pending_context.performance = performance

    def _pending_parts(self) -> list[dict]:
        return getattr(self._pending_context, "parts", [])

    def _pending_performance(self) -> list[dict]:
        return getattr(self._pending_context, "performance", [])

    @staticmethod
    def _base_row(
        item: dict,
        article_index: int,
        article_title,
        article_doi,
        fallback_order: int,
    ) -> dict:
        return {
            "article_index": article_index,
            "sample_index_in_article": (
                item.get("article_sample_order")
                or item.get("sample_index")
                or item.get("sample_no")
                or fallback_order
            ),
            "sample_name": item.get("sample_name"),
            "article_title": article_title,
            "DOI": article_doi,
        }

    def _rows_from_views(
        self,
        response: dict,
        article_index: int,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        title = response.get("article_title")
        doi = response.get("DOI")
        novelty = response.get("innovation")
        summary = response.get("ai_summary")
        entity_items = response.get("formulation_entities") or []
        part_items = response.get("entity_parts") or []
        performance_items = response.get("performance_samples") or response.get("performance") or []
        for name, items in (
            ("formulation_entities", entity_items),
            ("entity_parts", part_items),
            ("performance_samples", performance_items),
        ):
            if not isinstance(items, list):
                raise ValueError(f"One-shot JSON field {name!r} must be a list.")

        entity_rows = []
        for item_index, item in enumerate(entity_items, 1):
            if not isinstance(item, dict):
                raise ValueError(f"formulation_entities item {item_index} must be an object.")
            row = self._base_row(item, article_index, title, doi, item_index)
            row["novelty"] = novelty
            row["ai_literature_summary"] = summary
            for column, json_key in ENTITY_JSON_TO_COLUMN.items():
                row[column] = item.get(json_key)
            entity_rows.append(
                {column: eva.clean_value(row.get(column)) for column in ENTITY_OUTPUT_COLUMNS}
            )

        part_rows = []
        for item_index, item in enumerate(part_items, 1):
            if not isinstance(item, dict):
                raise ValueError(f"entity_parts item {item_index} must be an object.")
            row = self._base_row(item, article_index, title, doi, item_index)
            for column, json_key in PART_JSON_TO_COLUMN.items():
                row[column] = item.get(json_key)
            part_rows.append(
                {column: eva.clean_value(row.get(column)) for column in eva.PART_COLUMNS}
            )

        performance_rows = []
        record_counts: dict[tuple[str, str], int] = defaultdict(int)
        for item_index, item in enumerate(performance_items, 1):
            if not isinstance(item, dict):
                raise ValueError(f"performance_samples item {item_index} must be an object.")
            row = self._base_row(item, article_index, title, doi, item_index)
            row["novelty"] = novelty
            row["ai_literature_summary"] = summary
            row["formulation_alignment_status"] = "one_shot"
            for column, json_key in PERFORMANCE_JSON_TO_COLUMN.items():
                row[column] = item.get(json_key)
            record_key = (
                eva.normalize_key_number(row.get("sample_index_in_article")),
                eva.normalize_name(row.get("sample_name")),
            )
            record_counts[record_key] += 1
            row["performance_record_index"] = record_counts[record_key]
            row["performance_condition_signature"] = eva.build_performance_condition_signature(row)
            performance_rows.append(
                {column: eva.clean_value(row.get(column)) for column in eva.PERFORMANCE_COLUMNS}
            )

        return entity_rows, part_rows, performance_rows

    def parse_task_response(self, response_text: str, article_idx: int) -> list[dict]:
        response = json.loads(self._smart_fix_json(response_text))
        entity_rows, part_rows, performance_rows = self._rows_from_views(
            response,
            article_idx,
        )
        self._set_pending(part_rows, performance_rows)
        return entity_rows

    def append_to_excel(self, rows) -> None:
        eva.append_rows_to_excel(
            self.output_excel,
            rows,
            self.column_order,
            dedupe_exact=True,
        )
        eva.append_rows_to_excel(
            self.parts_excel,
            self._pending_parts(),
            eva.PART_COLUMNS,
            dedupe_exact=True,
        )
        eva.append_rows_to_excel(
            self.performance_excel,
            self._pending_performance(),
            eva.PERFORMANCE_COLUMNS,
            dedupe_exact=True,
        )
        self._set_pending([], [])

    def attach_source_metadata(
        self,
        rows: list[dict],
        file_id: str,
        paper_dir: str,
    ) -> list[dict]:
        rows = super().attach_source_metadata(rows, file_id, paper_dir)
        markdown_path = eva.source_md_path(paper_dir)
        source_folder = os.path.normpath(os.path.abspath(paper_dir))
        for row in self._pending_parts() + self._pending_performance():
            row["source_id"] = file_id
            row["source_folder"] = source_folder
            row["source_markdown_path"] = markdown_path
            eva.attach_stable_identity(row, file_id)
        return rows

    def _drop_rows_for_file(self, file_id: str) -> int:
        dropped = 0
        for path in (self.output_excel, self.parts_excel, self.performance_excel):
            if not os.path.exists(path):
                continue
            try:
                frame = eva.read_excel_canonical(path)
            except Exception:
                continue
            if "source_id" not in frame.columns:
                continue
            mask = frame["source_id"].astype(str).eq(str(file_id))
            count = int(mask.sum())
            if count:
                frame.loc[~mask].to_excel(path, index=False)
                dropped += count
        return dropped


class NaiveEntityFormulationAnalyzer(eva.EntityFormulationAnalyzer):
    def __init__(self, output_excel: str, parts_excel: str, model_name: str, input_folder: str):
        super().__init__(
            output_excel=output_excel,
            parts_excel=parts_excel,
            model_name=model_name,
            input_folder=input_folder,
        )
        self.prompt_template = SPLIT_NAIVE_FORMULATION_PROMPT


class NaivePerformanceAnalyzer(eva.PerformanceAnalyzer):
    def __init__(self, output_excel: str, model_name: str, input_folder: str):
        super().__init__(
            output_excel=output_excel,
            model_name=model_name,
            input_folder=input_folder,
        )
        self.prompt_template = SPLIT_NAIVE_PERFORMANCE_PROMPT


class FrozenEntityFormulationAnalyzer(eva.EntityFormulationAnalyzer):
    def __init__(self, output_excel: str, parts_excel: str, model_name: str, input_folder: str):
        super().__init__(
            output_excel=output_excel,
            parts_excel=parts_excel,
            model_name=model_name,
            input_folder=input_folder,
        )
        self.prompt_template = FROZEN_ENTITY_FORMULATION_PROMPT


class FrozenPerformanceAnalyzer(eva.PerformanceAnalyzer):
    def __init__(self, output_excel: str, model_name: str, input_folder: str):
        super().__init__(
            output_excel=output_excel,
            model_name=model_name,
            input_folder=input_folder,
        )
        self.prompt_template = FROZEN_PERFORMANCE_PROMPT


def run_one_shot(
    args: argparse.Namespace,
    articles: list[ArticleInput],
    output_dir: Path,
    configuration: str,
) -> None:
    prompt = (
        DIRECT_NAIVE_TRI_VIEW_PROMPT
        if configuration == "direct_naive_tri_view"
        else EXPERT_TRI_VIEW_SINGLE_SHOT_PROMPT
    )
    analyzer = TriViewSingleShotAnalyzer(
        output_excel=str(output_dir / "formulation_entities.xlsx"),
        parts_excel=str(output_dir / "formulation_entity_parts.xlsx"),
        performance_excel=str(output_dir / "performance_samples_final.xlsx"),
        model_name=args.model,
        input_folder=args.input,
        prompt_template=prompt,
    )
    analyzer.task_name = configuration
    configure_analyzer(analyzer, args)
    process_articles(analyzer, articles, args.workers)
    analyzer.summarize_run()


def run_split(
    args: argparse.Namespace,
    articles: list[ArticleInput],
    output_dir: Path,
    expert: bool,
) -> None:
    formulation_class = (
        FrozenEntityFormulationAnalyzer if expert else NaiveEntityFormulationAnalyzer
    )
    performance_class = FrozenPerformanceAnalyzer if expert else NaivePerformanceAnalyzer

    formulation = formulation_class(
        output_excel=str(output_dir / "formulation_entities.xlsx"),
        parts_excel=str(output_dir / "formulation_entity_parts.xlsx"),
        model_name=args.model,
        input_folder=args.input,
    )
    configure_analyzer(formulation, args)
    process_articles(formulation, articles, args.workers)
    formulation.summarize_run()

    performance = performance_class(
        output_excel=str(output_dir / "performance_samples.xlsx"),
        model_name=args.model,
        input_folder=args.input,
    )
    configure_analyzer(performance, args)
    process_articles(performance, articles, args.workers)
    performance.summarize_run()

    if expert and args.include_visual_stage:
        visual = eva.VisualPerformanceAnalyzer(
            output_excel=str(output_dir / "performance_samples_visual.xlsx"),
            model_name=args.model,
            input_folder=args.input,
        )
        configure_analyzer(visual, args)
        process_articles(visual, articles, args.workers)
        visual.summarize_run()

    if not args.dry_run:
        merge_args = copy_args(
            args,
            output_dir=str(output_dir),
            primary_performance=None,
            visual_performance=None,
            merged_performance=None,
            merged_wide=None,
        )
        eva.run_merge(merge_args)


def configurations_to_run(value: str) -> Iterable[str]:
    if value == "all":
        return PAPER_CONFIGURATIONS
    return (value,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the four paper GT50 extraction ablations."
    )
    parser.add_argument(
        "--configuration",
        choices=[*PAPER_CONFIGURATIONS, "all"],
        required=True,
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--gt-workbook", default=str(DEFAULT_GT_WORKBOOK))
    parser.add_argument("--input-order", default=str(DEFAULT_INPUT_ORDER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=eva.DEFAULT_MODEL)
    parser.add_argument("--api-url", default=eva.DEFAULT_API_URL)
    parser.add_argument("--api-keys-file", default=None)
    parser.add_argument("--api-key-limit", type=int, default=0)
    parser.add_argument("--force-vision", action="store_true")
    parser.add_argument("--include-visual-stage", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--api-retries", type=int, default=4)
    parser.add_argument("--retry-base-delay", type=float, default=20)
    parser.add_argument("--rate-limit-sleep", type=float, default=1800)
    parser.add_argument("--rate-limit-retries", type=int, default=3)
    parser.add_argument("--read-timeout", type=float, default=540)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input).resolve()
    output_root = Path(args.output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new path: {output_root}"
        )

    articles = load_gt50_articles(
        input_root=input_root,
        gt_workbook=Path(args.gt_workbook).resolve(),
        input_order_csv=Path(args.input_order).resolve(),
    )
    output_root.mkdir(parents=True)

    for configuration in configurations_to_run(args.configuration):
        output_dir = output_root / safe_name(args.model) / configuration
        output_dir.mkdir(parents=True)
        job_args = copy_args(args, output_dir=str(output_dir))
        print(
            f"Running configuration={configuration}, model={args.model}, "
            f"output={output_dir}"
        )
        if configuration in {
            "direct_naive_tri_view",
            "expert_tri_view_single_shot",
        }:
            run_one_shot(job_args, articles, output_dir, configuration)
        elif configuration == "split_naive":
            run_split(job_args, articles, output_dir, expert=False)
        elif configuration == "full_pipeline":
            run_split(job_args, articles, output_dir, expert=True)
        else:
            raise ValueError(f"Unknown configuration: {configuration}")


if __name__ == "__main__":
    main()
