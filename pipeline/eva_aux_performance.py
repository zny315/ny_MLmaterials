import argparse
import json
from pathlib import Path
from typing import Dict, List

import eva_split_pipeline as base


BASE_DIR = base.BASE_DIR
DEFAULT_INPUT_DIR = base.DEFAULT_INPUT_DIR
DEFAULT_OUTPUT_DIR = base.REPO_ROOT / "outputs" / "eva_auxiliary"
DEFAULT_API_KEYS_FILE = base.DEFAULT_API_KEYS_FILE
DEFAULT_MODEL = base.DEFAULT_MODEL
DEFAULT_API_URL = base.DEFAULT_API_URL


AUX_PERFORMANCE_COLUMNS = [
    "文章编号", "文章内样品序号", "样品名称", "文章名称", "DOI",
    *base.SOURCE_COLUMNS,
    "老化后抗拉强度", "老化后断裂伸长率",
    "体积电阻率", "介电强度",
    "熔体流动速率", "平衡扭矩", "热延伸", "凝胶含量", "交联密度",
    "硬度", "烟密度", "酸气释放", "氧化诱导时间", "氧化诱导温度",
    "性能备注", "创新点", "ai文献小结",
]


AUX_PERFORMANCE_PROMPT = """
You are a materials researcher building an auxiliary EVA/PE/POE cable-performance database.

Extract ONLY the non-core cable, processing, electrical, aging, and oxidation-stability performance data from the paper, including text, tables, captions, and charts.
This task is AUXILIARY PERFORMANCE ONLY. Do NOT extract LOI, UL-94, tensile strength, TGA temperatures, cone calorimetry, pHRR, THR, pSPR, TSP, pCOR, pCO2R, or residue metrics here.
Return valid JSON only. Do not output Markdown, explanations, or extra text.
Use "#" for missing values. Numeric fields must contain numeric values only.

[CRITICAL SAMPLE COVERAGE RULES]
1. Extract every tested sample that appears in the relevant auxiliary-performance tables, plots, or captions.
2. Preserve sample names exactly as written when no formulation roster is provided. If a formulation roster is provided, follow the formulation-guided alignment rules below.
3. If a sample is absent from a table, do NOT copy values from adjacent samples.
4. If exact values are only visible in a chart and can be read with reasonable confidence, extract the approximate numeric value and mention chart extraction in performance_notes.
5. If exact value is not readable, output "#".

[SOURCE SCOPE RULES - STRICT]
1. Prefer original experimental samples prepared or tested in this paper.
2. For review papers, literature-summary tables, introduction/background comparisons, or cited prior-work examples, extract auxiliary performance only when the source clearly reports sample-level values for an EVA/PE/POE-family material and the sample identity can be aligned. Otherwise output "#" for unavailable metrics rather than creating uncertain samples.
3. Do NOT treat method conditions, cited examples, mechanism labels, generic material names, or figure-legend labels as tested samples unless a sample-level value is explicitly reported.
4. If a paper compares several polymer families, keep only the values that belong to the exact sample label being output; do not transfer values from PC, ABS, PP, EPDM, PA, or other matrices to EVA/PE/POE samples.

[FORMULATION-GUIDED ALIGNMENT RULES - STRICT]
1. A "FORMULATION SAMPLE ROSTER" may be provided before the paper content. It is retrieved from the formulation task for this exact paper.
2. Use the roster as the alignment anchor. When an auxiliary-performance sample corresponds to a roster sample, output sample_name EXACTLY as the roster sample_name.
3. Return one JSON sample object for every roster sample. Fill unavailable auxiliary metrics with "#"; do not copy values from other samples.
4. If the paper reports auxiliary performance for an extra sample not present in the roster, also output it using the exact paper label and mention "not in formulation roster" in performance_notes.
5. If a performance table uses aliases, abbreviations, or shortened labels, map them to roster names only when the evidence is clear from table order, captions, or surrounding text. Mention the source alias in performance_notes.
6. Do not rely on row order alone unless the table explicitly lists the same ordered sample series as the roster.
7. If the same shortened performance label appears under different parent groups, do NOT output the shortened label alone. Include the parent group or map to the exact roster name.
8. If an alias could match more than one roster sample, do NOT guess. Output the paper label with the parent context in sample_name and explain the ambiguity in performance_notes.

[ANTI-HALLUCINATION]
Never synthesize missing metrics.
Never assume all samples were tested for all metrics.
Never use values from neat matrix or control samples for modified samples unless the sample names match exactly.
Never repair incompatible experiments or incompatible dimensions by guesswork.
Use "#" when uncertain.

JSON schema:
{
  "article_title": "paper title",
  "DOI": "paper DOI",
  "innovation": "paper innovation, concise",
  "ai_summary": "short paper summary",
  "samples": [
    {
      "sample_name": "Sample identifier/name. If a formulation roster is provided and alignment is clear, output the exact roster sample_name. Otherwise preserve the paper label and parent table context. Do not guess aliases when ambiguous.",
      "老化后抗拉强度": "Tensile strength after aging, numeric MPa only. Must be explicitly after thermal/weathering/aging treatment.",
      "老化后断裂伸长率": "Elongation at break after aging, numeric % only. Must be explicitly after aging treatment.",
      "体积电阻率": "Volume resistivity numeric value only when explicitly reported. Do NOT put surface resistivity, dielectric constant, dielectric loss, or conductivity unless it is clearly volume resistivity.",
      "介电强度": "Dielectric breakdown strength numeric value only when explicitly reported. Do NOT put dielectric constant, loss tangent, volume resistivity, or insulation resistance here.",
      "熔体流动速率": "MFI/MFR/melt flow index/rate, numeric g/10 min only. Do NOT put viscosity, torque, activation energy, power consumption, or processing temperature here.",
      "平衡扭矩": "Stable/final/equilibrium processing torque from Brabender/mixer/rheometer, numeric only. Mention original unit in performance_notes.",
      "热延伸": "Hot-set elongation/deformation, numeric % only. Do NOT put tensile elongation at break here.",
      "凝胶含量": "Gel content/fraction, numeric % only. Do NOT put char or TGA residue here.",
      "交联密度": "Crosslink density/network density numeric value only when explicitly reported. Mention unit/method in performance_notes.",
      "硬度": "Hardness numeric value only, usually Shore A or Shore D. Mention scale in performance_notes.",
      "烟密度": "Smoke density/optical smoke density/Ds/Dm numeric value only. Do NOT put TSP, pSPR, or light transmission here.",
      "酸气释放": "Acid gas emission/release numeric value only. Mention unit/method in performance_notes.",
      "氧化诱导时间": "Oxidation induction time/OIT, numeric min only when explicitly reported.",
      "氧化诱导温度": "Oxidation induction temperature/OITP, numeric deg C only when explicitly reported.",
      "性能备注": "Short notes for source table/figure, sample alias, test atmosphere, test condition, unit conversion, chart-read approximation, or ambiguity."
    }
  ]
}

Paper Content:
"""


class AuxPerformanceAnalyzer(base.PerformanceAnalyzer):
    task_name = "aux_performance"

    def __init__(self, output_excel: str, model_name: str, input_folder: str):
        super().__init__(output_excel=output_excel, model_name=model_name, input_folder=input_folder)
        self.prompt_template = AUX_PERFORMANCE_PROMPT
        self.column_order = AUX_PERFORMANCE_COLUMNS

    def parse_task_response(self, response_text: str, article_idx: int) -> List[Dict]:
        response = json.loads(self._smart_fix_json(response_text))
        article_title = response.get("article_title")
        article_doi = response.get("DOI")
        innovation = response.get("innovation")
        ai_summary = response.get("ai_summary")
        rows = []

        mapping = {
            "老化后抗拉强度": "老化后抗拉强度",
            "老化后断裂伸长率": "老化后断裂伸长率",
            "体积电阻率": "体积电阻率",
            "介电强度": "介电强度",
            "熔体流动速率": "熔体流动速率",
            "平衡扭矩": "平衡扭矩",
            "热延伸": "热延伸",
            "凝胶含量": "凝胶含量",
            "交联密度": "交联密度",
            "硬度": "硬度",
            "烟密度": "烟密度",
            "酸气释放": "酸气释放",
            "氧化诱导时间": "氧化诱导时间",
            "氧化诱导温度": "氧化诱导温度",
            "性能备注": "性能备注",
        }

        response_samples = response.get("samples")
        if not isinstance(response_samples, list):
            return []

        matched_roster_names = set()
        for sample_idx, sample in enumerate(response_samples, 1):
            raw_sample_name = sample.get("sample_name")
            aligned_name, aligned_order, matched_roster = self._align_to_formulation_roster(article_idx, raw_sample_name, sample_idx)
            if matched_roster:
                matched_roster_names.add(base.normalize_name(aligned_name))
                if base.normalize_name(raw_sample_name) != base.normalize_name(aligned_name):
                    notes = base.clean_value(sample.get("性能备注"))
                    alias_note = f"source performance label: {raw_sample_name}"
                    sample["性能备注"] = f"{notes}; {alias_note}" if notes else alias_note
            row = {
                "文章编号": article_idx,
                "文章内样品序号": aligned_order,
                "样品名称": aligned_name,
                "文章名称": article_title,
                "DOI": article_doi,
                "创新点": innovation,
                "ai文献小结": ai_summary,
            }
            for excel_col, json_key in mapping.items():
                row[excel_col] = sample.get(json_key)
            rows.append({col: base.clean_value(row.get(col)) for col in AUX_PERFORMANCE_COLUMNS})

        for roster_sample in self._formulation_samples_for_article(article_idx):
            roster_name = roster_sample.get("sample_name")
            roster_norm = base.normalize_name(roster_name)
            if not roster_norm or roster_norm in matched_roster_names:
                continue
            row = {
                "文章编号": article_idx,
                "文章内样品序号": roster_sample.get("article_sample_order"),
                "样品名称": roster_name,
                "文章名称": article_title,
                "DOI": article_doi,
                "性能备注": "No explicit auxiliary-performance value found for this formulation roster sample.",
                "创新点": innovation,
                "ai文献小结": ai_summary,
            }
            rows.append({col: base.clean_value(row.get(col)) for col in AUX_PERFORMANCE_COLUMNS})
        return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Extract auxiliary EVA/PE/POE performance fields only.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_DIR), help="MinerU paper folder. Default: inputs/mineru.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder. Default: outputs/eva.")
    parser.add_argument(
        "--formulation-excel",
        default=None,
        help="Optional formulation_entities.xlsx path used for sample alignment. Default: output-dir/formulation_entities.xlsx.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Chat model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"OpenAI-compatible chat completions endpoint. Default: {DEFAULT_API_URL}")
    parser.add_argument("--force-vision", action="store_true", help="Force sending vision-format image inputs even if model name does not look multimodal.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N paper folders. Default: 0 means all.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent paper requests per task. Default: 1.")
    parser.add_argument(
        "--api-keys-file",
        default=str(DEFAULT_API_KEYS_FILE) if DEFAULT_API_KEYS_FILE.exists() else None,
        help="Text file with one API key per line. Default: .secrets/api_keys.txt if it exists.",
    )
    parser.add_argument("--api-retries", type=int, default=4, help="API attempts per paper. Default: 4.")
    parser.add_argument("--retry-base-delay", type=float, default=20, help="Initial retry delay in seconds, doubled each attempt. Default: 20.")
    parser.add_argument("--rate-limit-sleep", type=float, default=1800, help="Seconds to wait after a rate-limit response. Default: 1800.")
    parser.add_argument("--rate-limit-retries", type=int, default=3, help="Rate-limit wait/retry cycles per paper/key. Default: 3.")
    parser.add_argument("--read-timeout", type=float, default=540, help="API read timeout in seconds. Default: 540.")
    parser.add_argument("--max-tokens", type=int, default=65536, help="Maximum output tokens per API request. Default: 65536.")
    parser.add_argument("--stream", action="store_true", help="Request streaming chat output and assemble streamed chunks. Helps avoid gateway timeout during long generation.")
    parser.add_argument("--dry-run", action="store_true", help="Generate API previews without calling the API.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing pipeline outputs in output-dir before running.")
    return parser.parse_args()


def clear_pipeline_outputs(output_dir: Path):
    output_files = [
        output_dir / "aux_performance_samples.xlsx",
        output_dir / "run.log",
    ]
    for path in output_files:
        if path.exists() and path.is_file():
            path.unlink()

    raw_dir = output_dir / "api_raw"
    if raw_dir.exists():
        for raw_file in raw_dir.glob("*.json"):
            if raw_file.is_file():
                raw_file.unlink()


def run_aux_performance(args):
    analyzer = AuxPerformanceAnalyzer(
        output_excel=str(Path(args.output_dir) / "aux_performance_samples.xlsx"),
        model_name=args.model,
        input_folder=args.input,
    )
    if args.formulation_excel:
        analyzer.formulation_excel = Path(args.formulation_excel)
    base.configure_runtime(analyzer, args, base.make_api_key_queue(base.load_api_keys(args.api_keys_file)))
    analyzer.process_folder(args.input)
    analyzer.summarize_run()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and not args.dry_run:
        clear_pipeline_outputs(output_dir)
    run_aux_performance(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
