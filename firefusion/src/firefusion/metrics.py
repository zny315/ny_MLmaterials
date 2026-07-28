from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"split_id", "target", "task_type", "scenario", "N", "trial", "method", "y_true", "y_pred"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"prediction table missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    keys = ["split_id", "target", "task_type", "scenario", "N", "trial", "method"]
    for key, group in predictions.groupby(keys, sort=True, dropna=False):
        split_id, target, task, scenario, n_shot, trial, method = key
        y = group["y_true"].to_numpy(float)
        p = group["y_pred"].to_numpy(float)
        common = dict(split_id=split_id, target=target, task_type=task, scenario=scenario,
                      N=int(n_shot), trial=int(trial), method=method, n_test=len(group))
        if task == "reg":
            rows.append({**common, "metric": "RMSE", "value": float(mean_squared_error(y, p) ** 0.5)})
            rows.append({**common, "metric": "R2", "value": float(r2_score(y, p))})
        else:
            value = float(roc_auc_score(y.astype(int), p)) if len(np.unique(y)) == 2 else np.nan
            rows.append({**common, "metric": "AUC", "value": value})
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["target", "task_type", "scenario", "N", "method", "metric"]
    return (
        metrics.groupby(keys, sort=True, dropna=False)["value"]
        .agg(mean="mean", std="std", trial_count="count")
        .reset_index()
    )
