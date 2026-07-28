from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


class EstimatorError(RuntimeError):
    pass


COMPACT_METHODS = {"ridge", "logistic", "gp", "tabpfn"}


def feature_view(method: str) -> str:
    return "compact" if method in COMPACT_METHODS else "full"


def positive_probability(estimator: Any, X: np.ndarray) -> np.ndarray:
    classes = list(estimator.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return np.asarray(estimator.predict_proba(X)[:, classes.index(1)], dtype=float)


def predict_estimator(
    method: str,
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    *,
    role: str,
    config: dict[str, Any],
    sample_weight: np.ndarray | None = None,
    tabpfn_device: str = "cuda:0",
) -> tuple[np.ndarray, str]:
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) == 0:
        raise EstimatorError("cannot fit on an empty training set")
    if method == "mean":
        return np.full(len(X_query), float(np.mean(y_train))), "fit"
    if np.all(y_train == y_train[0]) and (task == "clf" or method == "tabpfn"):
        return np.full(len(X_query), float(y_train[0])), "structural_constant_target"

    settings = config["estimators"]
    if method in COMPACT_METHODS:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_query = scaler.transform(X_query)
        if method == "tabpfn" and not np.any(X_train != X_train[0]):
            return np.full(len(X_query), float(np.mean(y_train))), "structural_constant_features_mean"

    if method == "ridge" and task == "reg":
        model = Ridge(alpha=float(settings["ridge_alpha"])).fit(X_train, y_train, sample_weight=sample_weight)
        return np.asarray(model.predict(X_query), float), "fit"
    if method == "logistic" and task == "clf":
        model = LogisticRegression(
            C=float(settings["logistic_C"]), max_iter=1000, solver="liblinear",
            class_weight="balanced", random_state=0,
        ).fit(X_train, y_train.astype(int), sample_weight=sample_weight)
        return positive_probability(model, X_query), "fit"
    if method == "rf":
        if task == "reg":
            model = RandomForestRegressor(
                n_estimators=int(settings["rf_trees"]), random_state=0, n_jobs=1,
            ).fit(X_train, y_train, sample_weight=sample_weight)
            return np.asarray(model.predict(X_query), float), "fit"
        model = RandomForestClassifier(
            n_estimators=int(settings["rf_trees"]), random_state=0, n_jobs=1,
            class_weight="balanced",
        ).fit(X_train, y_train.astype(int), sample_weight=sample_weight)
        return positive_probability(model, X_query), "fit"
    if method == "histgb":
        min_leaf = int(settings["lab_histgb_min_samples_leaf"]) if role == "lab" else 20
        if task == "reg":
            model = HistGradientBoostingRegressor(random_state=0, min_samples_leaf=min_leaf)
            model.fit(X_train, y_train, sample_weight=sample_weight)
            return np.asarray(model.predict(X_query), float), "fit"
        model = HistGradientBoostingClassifier(random_state=0, min_samples_leaf=min_leaf)
        model.fit(X_train, y_train.astype(int), sample_weight=sample_weight)
        return positive_probability(model, X_query), "fit"
    if method == "gp":
        if sample_weight is not None:
            raise EstimatorError("Gaussian-process baseline does not accept sample weights")
        if task == "reg":
            model = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(3.0) + WhiteKernel(0.1),
                normalize_y=True, alpha=1e-6,
            ).fit(X_train, y_train)
            return np.asarray(model.predict(X_query), float), "fit"
        model = GaussianProcessClassifier(kernel=ConstantKernel(1.0) * RBF(3.0)).fit(
            X_train, y_train.astype(int)
        )
        return positive_probability(model, X_query), "fit"
    if method == "tabpfn":
        if sample_weight is not None:
            raise EstimatorError("TabPFN baseline does not accept sample weights")
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError as exc:
            raise EstimatorError("TabPFN requested; install firefusion[tabpfn]") from exc
        kwargs: dict[str, Any] = {"device": tabpfn_device, "random_state": 0}
        if role == "literature":
            kwargs.update(n_estimators=8 if task == "reg" else 4,
                          n_jobs=1, fit_mode="fit_preprocessors")
        model_path = settings.get("tabpfn_model_path")
        if model_path:
            kwargs["model_path"] = model_path
        if task == "reg":
            model = TabPFNRegressor(**kwargs).fit(X_train, y_train)
            return np.asarray(model.predict(X_query), float), "fit"
        model = TabPFNClassifier(**kwargs).fit(X_train, y_train.astype(int))
        return positive_probability(model, X_query), "fit"
    raise EstimatorError(f"method {method!r} is incompatible with task {task!r}")


def predict_weighted_pool(
    X_source: np.ndarray,
    y_source: np.ndarray,
    source_weight: float,
    X_lab: np.ndarray,
    y_lab: np.ndarray,
    X_query: np.ndarray,
    *,
    config: dict[str, Any],
) -> np.ndarray:
    method = str(config["pooling"]["learner"])
    X = np.vstack([X_source, X_lab])
    y = np.concatenate([y_source, y_lab])
    weights = np.concatenate([
        np.full(len(y_source), float(source_weight), dtype=float),
        np.ones(len(y_lab), dtype=float),
    ])
    prediction, _ = predict_estimator(
        method, "reg", X, y, X_query, role="pool", config=config, sample_weight=weights
    )
    return prediction
