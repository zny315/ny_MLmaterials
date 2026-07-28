from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.special import expit
from scipy.stats import chi2, norm, qmc


class AffineError(RuntimeError):
    pass


@dataclass(frozen=True)
class PositiveAffineFit:
    intercept_standardized: float
    source_total_slope: float
    lab_coefficient: float
    literature_coefficient: float
    reference_center: float
    reference_scale: float
    posterior_df: float
    posterior_shape: float
    posterior_scale: float
    accepted_draw_count: int
    total_draw_count: int
    constraint_mass_qmc: float
    sobol_power_used: int
    untruncated_intercept: float
    untruncated_source_total_slope: float
    untruncated_lab_coefficient: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def reference_location_scale(reference: np.ndarray) -> tuple[float, float]:
    values = np.asarray(reference, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise AffineError("reference scores must be a finite vector with at least two rows")
    center = float(np.mean(values))
    scale = float(np.std(values, ddof=1))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise AffineError("reference score sample SD is zero or nonfinite")
    return center, scale


def standardize(values: np.ndarray, center: float, scale: float) -> np.ndarray:
    result = (np.asarray(values, dtype=np.float64) - center) / scale
    if not np.isfinite(result).all():
        raise AffineError("standardization produced nonfinite values")
    return result

def commensurability_power(
    observations: np.ndarray,
    source_predictions: np.ndarray,
    source_predictive_sigma: float,
    *,
    kappa: float,
) -> tuple[float, float, float]:
    y = np.asarray(observations, dtype=float)
    prediction = np.asarray(source_predictions, dtype=float)
    sigma = float(source_predictive_sigma)
    if y.ndim != 1 or y.shape != prediction.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise AffineError("commensurability inputs must be aligned finite vectors")
    if sigma <= np.finfo(float).eps or not np.isfinite(sigma):
        raise AffineError("source predictive sigma must be positive")
    if kappa <= 1.0 or not np.isfinite(kappa):
        raise AffineError("kappa must exceed one")
    sse = float(np.sum(np.square((y - prediction) / sigma)))
    log_bf = float(y.size * np.log(kappa) - 0.5 * (1.0 - 1.0 / kappa**2) * sse)
    return float(expit(log_bf)), log_bf, sse


@lru_cache(maxsize=None)
def _sobol_uniforms(power: int) -> np.ndarray:
    uniforms = qmc.Sobol(d=4, scramble=True, seed=0).random_base2(power)
    epsilon = np.finfo(np.float64).eps
    uniforms = np.clip(uniforms, epsilon, 1.0 - epsilon)
    uniforms.setflags(write=False)
    return uniforms


def fit_positive_affine(
    observations: np.ndarray,
    literature_train: np.ndarray,
    lab_loo_train: np.ndarray,
    reference_scores: np.ndarray,
    *,
    config: dict[str, Any],
) -> PositiveAffineFit:
    """Posterior mean under b>0 and 0<=c<=b.

    The standardized likelihood is
    y = beta0 + b*z_lit + c*(z_lab-z_lit) + epsilon.
    `prior_precision_diagonal` is K0 in the normal--inverse-gamma prior.
    """
    y = np.asarray(observations, dtype=np.float64)
    literature = np.asarray(literature_train, dtype=np.float64)
    lab_loo = np.asarray(lab_loo_train, dtype=np.float64)
    if y.ndim != 1 or y.size < 1 or not (y.shape == literature.shape == lab_loo.shape):
        raise AffineError("positive-affine inputs must be aligned nonempty vectors")
    if not (np.isfinite(y).all() and np.isfinite(literature).all() and np.isfinite(lab_loo).all()):
        raise AffineError("positive-affine inputs contain nonfinite values")
    center, scale = reference_location_scale(reference_scores)
    design = np.column_stack([
        np.ones(y.size),
        standardize(literature, center, scale),
        (lab_loo - literature) / scale,
    ])
    y_standard = standardize(y, center, scale)
    m0 = np.asarray(config["prior_mean"], dtype=float)
    precision_diagonal = np.asarray(config["prior_precision_diagonal"], dtype=float)
    if m0.shape != (3,) or precision_diagonal.shape != (3,) or np.any(precision_diagonal <= 0):
        raise AffineError("positive-affine prior vectors must have three valid entries")
    K0 = np.diag(precision_diagonal)
    Kn = K0 + design.T @ design
    mn = np.linalg.solve(Kn, K0 @ m0 + design.T @ y_standard)
    an = float(config["inverse_gamma_shape"] + 0.5 * y.size)
    bn = float(config["inverse_gamma_scale"] + 0.5 * (
        y_standard @ y_standard + m0 @ K0 @ m0 - mn @ Kn @ mn
    ))
    if not np.isfinite(bn) or bn <= 0:
        raise AffineError("invalid inverse-gamma posterior scale")
    shape = bn / an * np.linalg.inv(Kn)
    shape = 0.5 * (shape + shape.T)
    root = np.linalg.cholesky(shape)
    df = 2.0 * an
    first_power = int(config["sobol_power"])
    maximum_power = int(config.get("maximum_sobol_power", first_power))
    minimum = int(config["minimum_accepted_draws"])
    accepted_draws: np.ndarray | None = None
    accepted_count = 0
    total = 0
    power_used = first_power
    for power in range(first_power, maximum_power + 1):
        uniforms = _sobol_uniforms(power)
        normals = norm.ppf(uniforms[:, :3])
        chi = chi2.ppf(uniforms[:, 3], df)
        draws = mn + (normals @ root.T) * np.sqrt(df / chi)[:, None]
        accepted = (draws[:, 1] > 0) & (draws[:, 2] >= 0) & (draws[:, 2] <= draws[:, 1])
        accepted_count = int(accepted.sum())
        total = len(draws)
        power_used = power
        if accepted_count >= minimum:
            accepted_draws = draws[accepted]
            break
    if accepted_draws is None:
        raise AffineError(
            f"constrained posterior retained only {accepted_count} of {total} Sobol draws"
        )
    constrained = accepted_draws.mean(axis=0)
    intercept, b, c = map(float, constrained)
    return PositiveAffineFit(
        intercept_standardized=intercept,
        source_total_slope=b,
        lab_coefficient=c,
        literature_coefficient=b - c,
        reference_center=center,
        reference_scale=scale,
        posterior_df=df,
        posterior_shape=an,
        posterior_scale=bn,
        accepted_draw_count=accepted_count,
        total_draw_count=total,
        constraint_mass_qmc=float(accepted_count / total),
        sobol_power_used=power_used,
        untruncated_intercept=float(mn[0]),
        untruncated_source_total_slope=float(mn[1]),
        untruncated_lab_coefficient=float(mn[2]),
    )


def predict_positive_affine(
    fit: PositiveAffineFit,
    literature: np.ndarray,
    lab: np.ndarray,
) -> np.ndarray:
    literature = np.asarray(literature, dtype=float)
    lab = np.asarray(lab, dtype=float)
    if literature.shape != lab.shape:
        raise AffineError("affine prediction inputs must be aligned")
    z_lit = (literature - fit.reference_center) / fit.reference_scale
    direction = (lab - literature) / fit.reference_scale
    result = fit.reference_center + fit.reference_scale * (
        fit.intercept_standardized + fit.source_total_slope * z_lit
        + fit.lab_coefficient * direction
    )
    if not np.isfinite(result).all():
        raise AffineError("positive-affine prediction is nonfinite")
    return result
