from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

from .affine import (
    commensurability_power, fit_positive_affine, predict_positive_affine,
    reference_location_scale, standardize,
)
from .config import load_config
from .data import Snapshot
from .estimators import feature_view, predict_estimator, predict_weighted_pool
from .io import prepare_output, run_metadata, write_json, write_jsonl
from .metrics import evaluate_predictions, summarize_metrics
from .splits import SplitRecord, load_splits
from .synthesis import fit_synthesis, synthesize


@dataclass
class EndpointExperts:
    target: str
    snapshot: Snapshot
    truth: np.ndarray
    source_x_full: np.ndarray
    source_y: np.ndarray
    selected_source: np.ndarray
    source_histgb: np.ndarray
    source_sigma: float
    lab_method: str
    config: dict[str, Any]
    tabpfn_device: str

    def __post_init__(self) -> None:
        self.position = self.snapshot.target_position
        self.lab_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], np.ndarray] = {}
        self.pool_cache: dict[tuple[str, ...], tuple[np.ndarray, dict[str, Any]]] = {}
        self.affine_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], tuple[np.ndarray, dict[str, Any]]] = {}

    def _ordered(self, ids: Iterable[str], *, allow_empty: bool = False) -> tuple[str, ...]:
        result = tuple(str(value) for value in ids)
        if not allow_empty and not result:
            raise ValueError("empty sample-ID sequence")
        if len(result) != len(set(result)):
            raise ValueError("duplicate sample ID")
        unknown = [value for value in result if value not in self.position]
        if unknown:
            raise ValueError(f"unknown sample ID: {unknown[0]}")
        return result

    def _indices(self, ids: Iterable[str]) -> np.ndarray:
        return np.asarray([self.position[value] for value in ids], dtype=int)

    def lab_predictions(self, train_ids: Iterable[str], query_ids: Iterable[str]) -> np.ndarray:
        train = self._ordered(train_ids)
        query = self._ordered(query_ids)
        if set(train).intersection(query):
            raise ValueError("lab train/query overlap")
        key = (train, query)
        if key not in self.lab_cache:
            view = feature_view(self.lab_method)
            X = self.snapshot.target_X(view=view)
            prediction, _ = predict_estimator(
                self.lab_method, "reg", X[self._indices(train)], self.truth[self._indices(train)],
                X[self._indices(query)], role="lab", config=self.config,
                tabpfn_device=self.tabpfn_device,
            )
            self.lab_cache[key] = np.asarray(prediction, dtype=float)
        return self.lab_cache[key]

    def pool_predictions(self, train_ids: Iterable[str]) -> tuple[np.ndarray, dict[str, Any]]:
        ordered = self._ordered(train_ids)
        subset = tuple(sorted(ordered, key=self.position.__getitem__))
        if subset not in self.pool_cache:
            index = self._indices(subset)
            alpha, log_bf, sse = commensurability_power(
                self.truth[index], self.source_histgb[index], self.source_sigma,
                kappa=float(self.config["pooling"]["kappa"]),
            )
            prediction = predict_weighted_pool(
                self.source_x_full, self.source_y, alpha,
                self.snapshot.X_target_full[index], self.truth[index],
                self.snapshot.X_target_full, config=self.config,
            )
            diagnostic = {
                "train_sample_ids": list(subset), "alpha": alpha,
                "log_bayes_factor": log_bf, "standardized_residual_sse": sse,
                "source_predictive_sigma": self.source_sigma,
            }
            self.pool_cache[subset] = (prediction, diagnostic)
        return self.pool_cache[subset]

    def affine_predictions(
        self, train_ids: Iterable[str], query_ids: Iterable[str]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        train = self._ordered(train_ids)
        query = self._ordered(query_ids)
        if set(train).intersection(query):
            raise ValueError("affine train/query overlap")
        key = (train, query)
        if key in self.affine_cache:
            return self.affine_cache[key]
        index = self._indices(train)
        if len(train) == 1:
            lab_loo = self.selected_source[index].copy()
            inner = [{"heldout_sample_id": train[0], "policy": "zero_label_lab_direction"}]
        else:
            lab_loo = np.empty(len(train), dtype=float)
            inner = []
            for row, heldout in enumerate(train):
                reduced = tuple(value for value in train if value != heldout)
                lab_loo[row] = self.lab_predictions(reduced, (heldout,))[0]
                inner.append({"heldout_sample_id": heldout, "train_sample_ids": list(reduced),
                              "policy": "strict_nested_lab_refit", "prediction": float(lab_loo[row])})
        fit = fit_positive_affine(
            self.truth[index], self.selected_source[index], lab_loo, self.selected_source,
            config=self.config["positive_affine"],
        )
        query_index = self._indices(query)
        lab_query = self.lab_predictions(train, query)
        prediction = predict_positive_affine(fit, self.selected_source[query_index], lab_query)
        diagnostic = {
            "train_sample_ids": list(train), "query_sample_ids": list(query),
            "fit": fit.as_dict(), "inner_lab_loo": inner,
        }
        self.affine_cache[key] = (prediction, diagnostic)
        return prediction, diagnostic


def _build_endpoint_experts(
    target: str,
    snapshot: Snapshot,
    config: dict[str, Any],
    tabpfn_device: str,
) -> tuple[EndpointExperts, dict[str, Any]]:
    selection = config["selected_backbones"][target]
    literature_method = selection["literature"]
    source_view = feature_view(literature_method)
    X_source_selected, y_source_selected, _ = snapshot.source_endpoint(target, view=source_view)
    selected_source, _ = predict_estimator(
        literature_method, "reg", X_source_selected, y_source_selected,
        snapshot.target_X(view=source_view), role="literature", config=config,
        tabpfn_device=tabpfn_device,
    )
    X_source_full, source_y, source_ids = snapshot.source_endpoint(target, view="full")
    if len(source_y) < 5:
        raise ValueError(f"at least five eligible literature rows are required for {target}")
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    source_oof = cross_val_predict(
        HistGradientBoostingRegressor(random_state=0), X_source_full, source_y,
        cv=folds, n_jobs=1, method="predict",
    )
    source_sigma = float(np.sqrt(np.mean(np.square(source_y - source_oof))))
    if not np.isfinite(source_sigma) or source_sigma <= np.finfo(float).eps:
        raise ValueError(f"invalid HistGB OOF predictive scale for {target}")
    histgb = HistGradientBoostingRegressor(random_state=0).fit(X_source_full, source_y)
    source_histgb = np.asarray(histgb.predict(snapshot.X_target_full), dtype=float)
    center, scale = reference_location_scale(selected_source)
    experts = EndpointExperts(
        target=target, snapshot=snapshot, truth=snapshot.target_y(target),
        source_x_full=X_source_full, source_y=source_y,
        selected_source=np.asarray(selected_source, dtype=float),
        source_histgb=source_histgb, source_sigma=source_sigma,
        lab_method=selection["lab"], config=config, tabpfn_device=tabpfn_device,
    )
    invariant = {
        "target": target, "lab_backbone": selection["lab"],
        "literature_backbone": literature_method,
        "pooling_reference_model": "histgb", "pooling_learner": config["pooling"]["learner"],
        "eligible_literature_rows": len(source_y), "source_sample_ids": source_ids.tolist(),
        "source_predictive_sigma": source_sigma,
        "response_reference_center": center, "response_reference_sample_sd": scale,
    }
    return experts, invariant


def _run_split(record: SplitRecord, experts: EndpointExperts) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_ids, test_ids = record.train_sample_ids, record.test_sample_ids
    if record.N < 2:
        raise ValueError("nested LOO transfer requires N >= 2")
    train_index = experts._indices(train_ids)
    test_index = experts._indices(test_ids)
    pool_loo = np.empty(record.N, dtype=float)
    affine_loo = np.empty(record.N, dtype=float)
    loo_details = []
    for row, heldout in enumerate(train_ids):
        reduced = tuple(value for value in train_ids if value != heldout)
        pool_all, pool_diag = experts.pool_predictions(reduced)
        affine_value, affine_diag = experts.affine_predictions(reduced, (heldout,))
        pool_loo[row] = pool_all[experts.position[heldout]]
        affine_loo[row] = affine_value[0]
        loo_details.append({"heldout_sample_id": heldout, "pool": pool_diag, "affine": affine_diag})
    center, scale = reference_location_scale(experts.selected_source)
    fit = fit_synthesis(
        standardize(experts.truth[train_index], center, scale),
        standardize(pool_loo, center, scale),
        standardize(affine_loo, center, scale),
        config=experts.config["synthesis"],
    )
    lab_test = experts.lab_predictions(train_ids, test_ids)
    pool_all, pool_diag = experts.pool_predictions(train_ids)
    pool_test = pool_all[test_index]
    affine_test, affine_diag = experts.affine_predictions(train_ids, test_ids)
    final = synthesize(pool_test, affine_test, fit.affine_weight)
    methods = {
        f"lab/{experts.lab_method}": lab_test,
        f"literature/{experts.config['selected_backbones'][record.target]['literature']}": experts.selected_source[test_index],
        "transfer/pooling": pool_test,
        "transfer/positive_affine": affine_test,
        "transfer/bayesian_two_expert": final,
    }
    common = {
        "split_id": record.split_id, "formulation_split_id": record.formulation_split_id,
        "target": record.target, "task_type": "reg", "scenario": record.scenario,
        "N": record.N, "trial": record.trial,
    }
    rows = []
    for method, predictions in methods.items():
        for sample_id, truth, prediction in zip(
            test_ids, experts.truth[test_index], predictions, strict=True
        ):
            rows.append({**common, "sample_id": sample_id, "method": method,
                         "y_true": float(truth), "y_pred": float(prediction)})
    diagnostic = {
        **common, "train_sample_ids": list(train_ids), "test_sample_ids": list(test_ids),
        "synthesis": fit.as_dict(), "pool_full": pool_diag, "affine_full": affine_diag,
        "loo": loo_details,
    }
    return rows, diagnostic


def run_transfer(
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
    records = [record for record in load_splits(data_path, config)
               if record.target in config["selected_backbones"]]
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must lie in [0, shard_count)")
    records = [record for record in records if record.trial % shard_count == shard_index]
    if not records:
        raise ValueError("no transfer splits assigned to this shard")
    output = prepare_output(output_path, overwrite=overwrite)
    endpoint_experts = {}
    invariants = []
    for target in config["selected_backbones"]:
        endpoint_experts[target], invariant = _build_endpoint_experts(
            target, snapshot, config, tabpfn_device
        )
        invariants.append(invariant)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for record in records:
        split_rows, split_diagnostic = _run_split(record, endpoint_experts[record.target])
        rows.extend(split_rows)
        diagnostics.append(split_diagnostic)
    predictions = pd.DataFrame(rows)
    metrics = evaluate_predictions(predictions)
    summary = summarize_metrics(metrics)
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "split_metrics.csv", index=False)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    write_jsonl(diagnostics, output / "diagnostics.jsonl")
    write_json(invariants, output / "endpoint_invariants.json")
    metadata = run_metadata(config_path, data_path, "transfer")
    metadata.update({"prediction_rows": len(predictions), "split_metric_rows": len(metrics),
                     "shard_index": shard_index, "shard_count": shard_count,
                     "shard_rule": "trial modulo shard_count"})
    write_json(metadata, output / "run_metadata.json")
    return output
