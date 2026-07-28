from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .io import prepare_output, write_json


def summarize_runs(
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = prepare_output(output_path, overwrite=overwrite)
    frames = []
    sources = []
    for input_path in input_paths:
        input_path = Path(input_path)
        metrics_path = input_path / "split_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        frame = pd.read_csv(metrics_path)
        frame.insert(0, "analysis", input_path.name)
        frames.append(frame)
        sources.append(str(input_path.resolve()))
    combined = pd.concat(frames, ignore_index=True)
    keys = ["analysis", "target", "task_type", "scenario", "N", "method", "metric"]
    summary = (
        combined.groupby(keys, sort=True, dropna=False)["value"]
        .agg(mean="mean", std="std", trial_count="count")
        .reset_index()
    )
    combined.to_csv(output / "combined_split_metrics.csv", index=False)
    summary.to_csv(output / "combined_summary_metrics.csv", index=False)
    write_json({"input_directories": sources, "split_metric_rows": len(combined)},
               output / "summary_metadata.json")
    return output
