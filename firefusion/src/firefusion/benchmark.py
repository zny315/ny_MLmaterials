from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .data import Snapshot
from .estimators import feature_view, predict_estimator
from .io import prepare_output, run_metadata, write_json
from .metrics import evaluate_predictions, summarize_metrics
from .splits import load_splits


def _source_class_weights(y: np.ndarray) -> np.ndarray:
    prevalence = float(np.mean(y.astype(int)))
    return np.where(
        y.astype(int) == 1,
        0.5 / max(prevalence, 1e-6),
        0.5 / max(1.0 - prevalence, 1e-6),
    )


def run_benchmark(
    config_path: str | Path,
    data_path: str | Path,
    output_path: str | Path,
    *,
    tabpfn_device: str = "cuda:0",
    shard_index: int = 0,
    shard_count: int = 1,
    overwrite: bool = False,
) -> Path:
    config_path, data_path = Path(config_path), Path(data_path)
    config = load_config(config_path)
    snapshot = Snapshot.load(data_path, config["targets"])
    records = load_splits(data_path, config)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    records = [record for record in records if record.trial % shard_count == shard_index]
    if not records:
        raise ValueError("no benchmark splits assigned to this shard")
    output = prepare_output(output_path, overwrite=overwrite)

    source_predictions: dict[tuple[str, str], np.ndarray] = {}
    source_status: dict[tuple[str, str], str] = {}
    for target, task in config["targets"].items():
        methods = config["benchmark"]["regression_methods" if task == "reg" else "classification_methods"]
        for method in methods:
            if method == "gp" and target not in config["benchmark"].get("gp_literature_targets", []):
                continue
            view = feature_view(method)
            X_source, y_source, _ = snapshot.source_endpoint(target, view=view)
            X_target = snapshot.target_X(view=view)
            weights = _source_class_weights(y_source) if task == "clf" and method == "histgb" else None
            prediction, status = predict_estimator(
                method, task, X_source, y_source, X_target,
                role="literature", config=config, sample_weight=weights,
                tabpfn_device=tabpfn_device,
            )
            source_predictions[(target, method)] = prediction
            source_status[(target, method)] = status

    rows: list[dict[str, Any]] = []
    for record in records:
        target, task = record.target, record.task_type
        y_all = snapshot.target_y(target)
        train_index = snapshot.positions(record.train_sample_ids)
        test_index = snapshot.positions(record.test_sample_ids)
        methods = config["benchmark"]["regression_methods" if task == "reg" else "classification_methods"]
        common = {
            "split_id": record.split_id,
            "formulation_split_id": record.formulation_split_id,
            "target": target,
            "task_type": task,
            "scenario": record.scenario,
            "N": record.N,
            "trial": record.trial,
        }
        for method in methods:
            view = feature_view(method)
            X_target = snapshot.target_X(view=view)
            prediction, status = predict_estimator(
                method, task, X_target[train_index], y_all[train_index], X_target[test_index],
                role="lab", config=config, tabpfn_device=tabpfn_device,
            )
            for sample_id, truth, value in zip(record.test_sample_ids, y_all[test_index], prediction, strict=True):
                rows.append({**common, "sample_id": sample_id, "method": f"lab/{method}",
                             "fit_status": status, "y_true": float(truth), "y_pred": float(value)})
            if (target, method) not in source_predictions:
                continue
            prediction = source_predictions[(target, method)][test_index]
            status = source_status[(target, method)]
            for sample_id, truth, value in zip(record.test_sample_ids, y_all[test_index], prediction, strict=True):
                rows.append({**common, "sample_id": sample_id, "method": f"literature/{method}",
                             "fit_status": status, "y_true": float(truth), "y_pred": float(value)})

    predictions = pd.DataFrame(rows)
    metrics = evaluate_predictions(predictions)
    summary = summarize_metrics(metrics)
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "split_metrics.csv", index=False)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    metadata = run_metadata(config_path, data_path, "benchmark")
    metadata.update({"prediction_rows": len(predictions), "split_metric_rows": len(metrics),
                     "shard_index": shard_index, "shard_count": shard_count,
                     "shard_rule": "trial modulo shard_count"})
    write_json(metadata, output / "run_metadata.json")
    return output
