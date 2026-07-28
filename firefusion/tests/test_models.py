import numpy as np

from firefusion.affine import fit_positive_affine, predict_positive_affine
from firefusion.synthesis import fit_synthesis, synthesize


def test_positive_affine_constraints_and_finite_prediction():
    literature = np.array([-0.5, 0.0, 0.8, 1.2, 1.8])
    lab = np.array([-0.3, 0.2, 0.9, 1.4, 2.0])
    y = np.array([-0.2, 0.1, 1.0, 1.3, 2.1])
    config = {
        "prior_mean": [0.0, 1.0, 0.0],
        "prior_precision_diagonal": [0.25, 4.0, 4.0],
        "inverse_gamma_shape": 2.0, "inverse_gamma_scale": 1.0,
        "sobol_power": 12, "minimum_accepted_draws": 32,
        "maximum_sobol_power": 16,
    }
    fit = fit_positive_affine(y, literature, lab, literature, config=config)
    assert fit.source_total_slope > 0
    assert 0 <= fit.lab_coefficient <= fit.source_total_slope
    assert fit.literature_coefficient >= 0
    assert np.isfinite(predict_positive_affine(fit, literature, lab)).all()


def test_synthesis_is_convex_and_audited():
    y = np.array([0.0, 0.4, 0.9, 1.2])
    pool = np.array([0.1, 0.3, 0.8, 1.0])
    affine = np.array([-0.1, 0.5, 1.0, 1.3])
    config = {
        "inverse_gamma_shape": 2.0, "inverse_gamma_scale": 1.0,
        "primary_quadrature_points": 64, "audit_quadrature_points": 128,
        "convergence_tolerance": 1e-10,
    }
    fit = fit_synthesis(y, pool, affine, config=config)
    prediction = synthesize(pool, affine, fit.affine_weight)
    assert fit.convergence_pass and 0 <= fit.affine_weight <= 1
    assert np.all(prediction >= np.minimum(pool, affine))
    assert np.all(prediction <= np.maximum(pool, affine))
