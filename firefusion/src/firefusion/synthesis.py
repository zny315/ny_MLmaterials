from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


class SynthesisError(ValueError):
    pass


@dataclass(frozen=True)
class SynthesisFit:
    affine_weight: float
    pool_weight: float
    audit_affine_weight: float
    quadrature_absolute_difference: float
    posterior_affine_weight_variance: float
    audit_posterior_affine_weight_variance: float
    primary_quadrature_points: int
    audit_quadrature_points: int
    observation_count: int
    degenerate_experts: bool
    convergence_pass: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _moments(
    y: np.ndarray,
    pool: np.ndarray,
    affine: np.ndarray,
    *,
    shape: float,
    scale: float,
    points: int,
) -> tuple[float, float]:
    nodes, weights = np.polynomial.legendre.leggauss(points)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    residual = y[:, None] - (pool[:, None] + (affine - pool)[:, None] * nodes[None, :])
    sum_squares = np.sum(residual**2, axis=0)
    log_mass = np.log(weights) - (shape + 0.5 * y.size) * np.log(scale + 0.5 * sum_squares)
    mass = np.exp(log_mass - np.max(log_mass))
    denominator = float(mass.sum())
    mean = float(np.sum(mass * nodes) / denominator)
    second = float(np.sum(mass * nodes**2) / denominator)
    return mean, max(0.0, second - mean**2)


def fit_synthesis(
    y_standardized: np.ndarray,
    pool_standardized: np.ndarray,
    affine_standardized: np.ndarray,
    *,
    config: dict[str, Any],
) -> SynthesisFit:
    y = np.asarray(y_standardized, dtype=float)
    pool = np.asarray(pool_standardized, dtype=float)
    affine = np.asarray(affine_standardized, dtype=float)
    if y.ndim != 1 or y.size == 0 or not (y.shape == pool.shape == affine.shape):
        raise SynthesisError("synthesis inputs must be aligned nonempty vectors")
    if not (np.isfinite(y).all() and np.isfinite(pool).all() and np.isfinite(affine).all()):
        raise SynthesisError("synthesis inputs contain nonfinite values")
    primary_points = int(config["primary_quadrature_points"])
    audit_points = int(config["audit_quadrature_points"])
    if audit_points <= primary_points:
        raise SynthesisError("audit quadrature must use more points than primary")
    degenerate = bool(np.array_equal(pool, affine))
    if degenerate:
        mean = audit_mean = 0.5
        variance = audit_variance = 1.0 / 12.0
    else:
        kwargs = dict(shape=float(config["inverse_gamma_shape"]),
                      scale=float(config["inverse_gamma_scale"]))
        mean, variance = _moments(y, pool, affine, points=primary_points, **kwargs)
        audit_mean, audit_variance = _moments(y, pool, affine, points=audit_points, **kwargs)
    difference = abs(mean - audit_mean)
    tolerance = float(config["convergence_tolerance"])
    if difference > tolerance:
        raise SynthesisError(f"quadrature audit failed: {difference} > {tolerance}")
    return SynthesisFit(
        affine_weight=mean, pool_weight=1.0 - mean,
        audit_affine_weight=audit_mean, quadrature_absolute_difference=difference,
        posterior_affine_weight_variance=variance,
        audit_posterior_affine_weight_variance=audit_variance,
        primary_quadrature_points=primary_points,
        audit_quadrature_points=audit_points,
        observation_count=len(y), degenerate_experts=degenerate,
        convergence_pass=True,
    )


def synthesize(pool: np.ndarray, affine: np.ndarray, affine_weight: float) -> np.ndarray:
    pool = np.asarray(pool, dtype=float)
    affine = np.asarray(affine, dtype=float)
    if pool.shape != affine.shape or not 0.0 <= affine_weight <= 1.0:
        raise SynthesisError("invalid synthesis prediction inputs")
    result = (1.0 - affine_weight) * pool + affine_weight * affine
    if not np.isfinite(result).all():
        raise SynthesisError("synthesis prediction is nonfinite")
    return result
