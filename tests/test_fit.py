import numpy as np

from mumag2022 import (
    ExperimentGeometry,
    fit_at_exchange_stiffness,
    fit_exchange_and_demagnetizing_field,
    fit_exchange_stiffness,
    generate_synthetic_series,
    stability_map,
)


def test_direct_nnls_recovers_noiseless_linear_spectra():
    series = generate_synthetic_series(relative_noise=1.0e-7, seed=3)
    result = fit_at_exchange_stiffness(
        series, 6.5, ExperimentGeometry.PERPENDICULAR
    )
    assert result.coefficients.shape == (3, series.q.size)
    assert result.mean_chi_square < 5.0
    assert np.all(result.rank == 3)
    assert np.all(np.isfinite(result.condition_number))


def test_exchange_profile_recovers_synthetic_value():
    series = generate_synthetic_series(relative_noise=0.01, seed=12)
    conditional = fit_at_exchange_stiffness(
        series, 6.5, ExperimentGeometry.PERPENDICULAR
    )
    result = fit_exchange_stiffness(
        series,
        ExperimentGeometry.PERPENDICULAR,
        exchange_bounds_pJ_per_m=(2.0, 12.0),
        profile_samples=41,
    )
    assert abs(result.best_fit.exchange_stiffness_pJ_per_m - 6.5) < 0.8
    assert result.profile_chi_square.min() == result.best_fit.chi_square
    assert result.characteristic_q_max_nm_inv > 0.0
    assert result.best_fit.degrees_of_freedom == conditional.degrees_of_freedom - 1


def test_stability_map_has_expected_shape_and_finite_centre():
    series = generate_synthetic_series(relative_noise=0.01, seed=4)
    result = stability_map(
        series,
        ExperimentGeometry.PERPENDICULAR,
        q_max_values=[0.20, 0.28],
        minimum_field_values_mT=[60.0, 100.0],
        exchange_bounds_pJ_per_m=(2.0, 12.0),
        profile_samples=15,
    )
    assert result.exchange_stiffness_pJ_per_m.shape == (2, 2)
    assert np.any(np.isfinite(result.exchange_stiffness_pJ_per_m))


def test_joint_demag_fit_returns_valid_surface():
    series = generate_synthetic_series(relative_noise=0.005, seed=5)
    result = fit_exchange_and_demagnetizing_field(
        series,
        ExperimentGeometry.PERPENDICULAR,
        exchange_bounds_pJ_per_m=(4.0, 9.0),
        demagnetizing_bounds_mT=(10.0, 30.0),
        exchange_samples=7,
        demagnetizing_samples=6,
    )
    assert result.chi_square_surface.shape == (6, 7)
    assert 4.0 <= result.best_fit.exchange_stiffness_pJ_per_m <= 9.0
    assert 10.0 <= result.best_demagnetizing_field_mT <= 30.0
