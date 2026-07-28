from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark" / "gt50"
EXPECTED = BENCHMARK / "expected_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay and verify frozen GT50 paper metrics.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "gt50_replay")
    return parser.parse_args()


def verify(summary_path: Path) -> None:
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    actual = pd.read_csv(summary_path).set_index(["method", "view"])
    tolerance = float(expected["float_tolerance"])
    errors = []
    for row in expected["metrics"]:
        key = (row["method"], row["view"])
        if key not in actual.index:
            errors.append(f"missing metric row: {key}")
            continue
        observed = actual.loc[key]
        for field in ("TP", "FP", "FN"):
            if int(observed[field]) != int(row[field]):
                errors.append(f"{key} {field}: {observed[field]} != {row[field]}")
        for field in ("precision", "recall", "F1"):
            if abs(float(observed[field]) - float(row[field])) > tolerance:
                errors.append(f"{key} {field}: {observed[field]} != {row[field]}")
    if errors:
        raise RuntimeError("GT50 replay mismatch:\n- " + "\n- ".join(errors))


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    command = [
        sys.executable,
        str(BENCHMARK / "evaluator" / "evaluate_gt50.py"),
        "--gt", str(BENCHMARK / "ground_truth" / "GT50_EVA_hierarchical_ground_truth_publication_clean.xlsx"),
        "--direct-dir", str(BENCHMARK / "extraction_outputs" / "direct_llm"),
        "--expert-dir", str(BENCHMARK / "extraction_outputs" / "expert_prompt_only"),
        "--split-dir", str(BENCHMARK / "extraction_outputs" / "task_decomposition_only"),
        "--full-dir", str(BENCHMARK / "extraction_outputs" / "full_pipeline"),
        "--output-dir", str(output_dir),
    ]
    subprocess.run(command, check=True)
    verify(output_dir / "metric_summary_candidate.csv")
    print("GT50 replay matches all expected paper metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
