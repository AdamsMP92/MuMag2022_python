"""Stable weighted non-negative fits for the MuMag2022 model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize, minimize_scalar, nnls

from .data import ExperimentGeometry, FieldSeries, PreparedSeries, prepare_series
from .response import characteristic_q_max_nm_inv, response_functions


@dataclass(frozen=True)
class FixedExchangeFit:
    """Complete linear solution conditional on one exchange stiffness."""

    geometry: ExperimentGeometry
    exchange_stiffness_pJ_per_m: float
    q: NDArray[np.float64]
    applied_fields_mT: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    conditional_standard_errors: NDArray[np.float64]
    response_h: NDArray[np.float64]
    response_m: NDArray[np.float64] | None
    fitted_intensity: NDArray[np.float64]
    residuals: NDArray[np.float64]
    standardized_residuals: NDArray[np.float64]
    chi_square: float
    mean_chi_square: float
    degrees_of_freedom: int
    reduced_chi_square: float
    condition_number: NDArray[np.float64]
    rank: NDArray[np.int64]
    active_constraints: NDArray[np.bool_]

    @property
    def parameter_names(self) -> tuple[str, ...]:
        if self.geometry is ExperimentGeometry.PERPENDICULAR:
            return ("I_res", "S_H", "S_M")
        return ("I_res", "S_H")

    @property
    def residual_intensity(self) -> NDArray[np.float64]:
        return self.coefficients[0]

    @property
    def anisotropy_scattering(self) -> NDArray[np.float64]:
        return self.coefficients[1]

    @property
    def magnetostatic_scattering(self) -> NDArray[np.float64] | None:
        if self.geometry is ExperimentGeometry.PERPENDICULAR:
            return self.coefficients[2]
        return None


@dataclass(frozen=True)
class ProfileInterval:
    """Approximate profile interval for one scalar parameter."""

    level: float
    threshold: float
    lower: float | None
    upper: float | None

    @property
    def complete(self) -> bool:
        return self.lower is not None and self.upper is not None


@dataclass(frozen=True)
class ExchangeFitResult:
    """Best fit and diagnostics for the exchange-stiffness profile."""

    best_fit: FixedExchangeFit
    profile_exchange_stiffness_pJ_per_m: NDArray[np.float64]
    profile_chi_square: NDArray[np.float64]
    curvature_standard_error_pJ_per_m: float | None
    interval_68: ProfileInterval
    interval_95: ProfileInterval
    characteristic_q_max_nm_inv: float
    q_range: tuple[float, float]
    minimum_applied_field_mT: float
    exchange_bounds_pJ_per_m: tuple[float, float]


@dataclass(frozen=True)
class StabilityMap:
    """Dependence of best-fit A on q_max and minimum applied field."""

    q_max_values: NDArray[np.float64]
    minimum_field_values_mT: NDArray[np.float64]
    exchange_stiffness_pJ_per_m: NDArray[np.float64]
    mean_chi_square: NDArray[np.float64]


@dataclass(frozen=True)
class DemagnetizingFitResult:
    """Joint A and constant demagnetizing-field profile."""

    best_fit: FixedExchangeFit
    best_demagnetizing_field_mT: float
    exchange_values_pJ_per_m: NDArray[np.float64]
    demagnetizing_values_mT: NDArray[np.float64]
    chi_square_surface: NDArray[np.float64]


def fit_at_exchange_stiffness(
    series: FieldSeries,
    exchange_stiffness_pJ_per_m: float,
    geometry: ExperimentGeometry | str,
    *,
    q_range: tuple[float, float] | None = None,
    minimum_applied_field_mT: float | None = None,
    demagnetizing_field_override_mT: float | None = None,
) -> FixedExchangeFit:
    """Solve the weighted NNLS problems directly at every selected q."""
    geometry = ExperimentGeometry(geometry)
    prepared = prepare_series(
        series,
        q_range=q_range,
        minimum_applied_field_mT=minimum_applied_field_mT,
        demagnetizing_field_override_mT=demagnetizing_field_override_mT,
    )
    parameter_count = 3 if geometry is ExperimentGeometry.PERPENDICULAR else 2
    if prepared.intensity.shape[0] < parameter_count:
        raise ValueError(
            f"{geometry.value} geometry needs at least {parameter_count} field curves"
        )

    response_h, response_m = response_functions(
        prepared.q,
        prepared.applied_fields_mT,
        prepared.saturation_magnetizations_mT,
        prepared.demagnetizing_fields_mT,
        exchange_stiffness_pJ_per_m,
        geometry,
    )
    return _fit_prepared(
        prepared,
        geometry,
        float(exchange_stiffness_pJ_per_m),
        response_h,
        response_m,
    )


def fit_exchange_stiffness(
    series: FieldSeries,
    geometry: ExperimentGeometry | str,
    *,
    exchange_bounds_pJ_per_m: tuple[float, float] = (0.1, 30.0),
    profile_samples: int = 81,
    q_range: tuple[float, float] | None = None,
    minimum_applied_field_mT: float | None = None,
) -> ExchangeFitResult:
    """Profile and refine the global exchange stiffness."""
    geometry = ExperimentGeometry(geometry)
    lower, upper = (float(value) for value in exchange_bounds_pJ_per_m)
    if not 0.0 < lower < upper:
        raise ValueError("exchange bounds must satisfy 0 < A_min < A_max")
    if profile_samples < 9:
        raise ValueError("profile_samples must be at least 9")
    selected = prepare_series(
        series,
        q_range=q_range,
        minimum_applied_field_mT=minimum_applied_field_mT,
    )
    q_limits = (float(selected.q[0]), float(selected.q[-1]))
    minimum_field = float(np.min(selected.applied_fields_mT))

    def objective(exchange: float) -> float:
        return fit_at_exchange_stiffness(
            series,
            exchange,
            geometry,
            q_range=q_limits,
            minimum_applied_field_mT=minimum_field,
        ).chi_square

    profile_a = np.linspace(lower, upper, int(profile_samples))
    profile_chi = np.asarray([objective(value) for value in profile_a])
    grid_minimum = int(np.nanargmin(profile_chi))
    if grid_minimum in {0, profile_a.size - 1}:
        best_a = float(profile_a[grid_minimum])
    else:
        optimum = minimize_scalar(
            objective,
            bounds=(profile_a[grid_minimum - 1], profile_a[grid_minimum + 1]),
            method="bounded",
            options={"xatol": max(1.0e-8, (upper - lower) * 1.0e-8)},
        )
        if not optimum.success or not np.isfinite(optimum.fun):
            raise RuntimeError(
                f"exchange-stiffness optimization failed: {optimum.message}"
            )
        best_a = float(optimum.x)
    best_fit = fit_at_exchange_stiffness(
        series,
        best_a,
        geometry,
        q_range=q_limits,
        minimum_applied_field_mT=minimum_field,
    )
    # ``fit_at_exchange_stiffness`` reports the conditional residual degrees
    # of freedom for fixed A.  The profiled result has estimated one additional
    # global nonlinear parameter and therefore loses one further degree of
    # freedom.
    global_degrees = max(best_fit.degrees_of_freedom - 1, 0)
    best_fit = replace(
        best_fit,
        degrees_of_freedom=global_degrees,
        reduced_chi_square=(
            best_fit.chi_square / global_degrees if global_degrees > 0 else np.nan
        ),
    )

    profile_a, profile_chi = _insert_profile_point(
        profile_a, profile_chi, best_a, best_fit.chi_square
    )
    interval_68 = _profile_interval(profile_a, profile_chi, best_a, 1.0, 0.6827)
    interval_95 = _profile_interval(profile_a, profile_chi, best_a, 3.8414588, 0.95)
    curvature = _curvature_standard_error(objective, best_a, lower, upper)
    q_characteristic = characteristic_q_max_nm_inv(
        best_a,
        float(np.mean(selected.saturation_magnetizations_mT)),
        float(np.max(selected.applied_fields_mT)),
    )
    return ExchangeFitResult(
        best_fit=best_fit,
        profile_exchange_stiffness_pJ_per_m=profile_a,
        profile_chi_square=profile_chi,
        curvature_standard_error_pJ_per_m=curvature,
        interval_68=interval_68,
        interval_95=interval_95,
        characteristic_q_max_nm_inv=q_characteristic,
        q_range=q_limits,
        minimum_applied_field_mT=minimum_field,
        exchange_bounds_pJ_per_m=(lower, upper),
    )


def stability_map(
    series: FieldSeries,
    geometry: ExperimentGeometry | str,
    *,
    q_max_values: ArrayLike,
    minimum_field_values_mT: ArrayLike,
    exchange_bounds_pJ_per_m: tuple[float, float] = (0.1, 30.0),
    q_min: float | None = None,
    profile_samples: int = 31,
) -> StabilityMap:
    """Repeat the fit over a transparent q_max/H_min grid."""
    q_max_values = np.asarray(q_max_values, dtype=float)
    field_values = np.asarray(minimum_field_values_mT, dtype=float)
    if q_min is None:
        q_min = float(series.q[0])
    exchange = np.full((field_values.size, q_max_values.size), np.nan)
    goodness = np.full_like(exchange, np.nan)
    for row, minimum_field in enumerate(field_values):
        for column, q_max in enumerate(q_max_values):
            try:
                result = fit_exchange_stiffness(
                    series,
                    geometry,
                    exchange_bounds_pJ_per_m=exchange_bounds_pJ_per_m,
                    profile_samples=profile_samples,
                    q_range=(float(q_min), float(q_max)),
                    minimum_applied_field_mT=float(minimum_field),
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
            exchange[row, column] = result.best_fit.exchange_stiffness_pJ_per_m
            goodness[row, column] = result.best_fit.mean_chi_square
    return StabilityMap(q_max_values, field_values, exchange, goodness)


def fit_exchange_and_demagnetizing_field(
    series: FieldSeries,
    geometry: ExperimentGeometry | str,
    *,
    exchange_bounds_pJ_per_m: tuple[float, float],
    demagnetizing_bounds_mT: tuple[float, float],
    q_range: tuple[float, float] | None = None,
    minimum_applied_field_mT: float | None = None,
    exchange_samples: int = 31,
    demagnetizing_samples: int = 31,
) -> DemagnetizingFitResult:
    """Fit A and one field-independent demagnetizing field."""
    geometry = ExperimentGeometry(geometry)
    a_values = np.linspace(*exchange_bounds_pJ_per_m, int(exchange_samples))
    hd_values = np.linspace(*demagnetizing_bounds_mT, int(demagnetizing_samples))
    if np.any(a_values <= 0.0) or hd_values[0] < 0.0:
        raise ValueError("A must be positive and Hd non-negative")
    surface = np.full((hd_values.size, a_values.size), np.nan)

    def objective(vector: NDArray[np.float64]) -> float:
        exchange, demag = (float(value) for value in vector)
        try:
            return fit_at_exchange_stiffness(
                series,
                exchange,
                geometry,
                q_range=q_range,
                minimum_applied_field_mT=minimum_applied_field_mT,
                demagnetizing_field_override_mT=demag,
            ).chi_square
        except ValueError:
            return np.inf

    for row, demag in enumerate(hd_values):
        for column, exchange in enumerate(a_values):
            surface[row, column] = objective(np.asarray([exchange, demag]))
    if not np.any(np.isfinite(surface)):
        raise ValueError("no valid A/Hd combination remains inside the selected data")
    row, column = np.unravel_index(np.nanargmin(surface), surface.shape)
    optimum = minimize(
        objective,
        x0=np.asarray([a_values[column], hd_values[row]]),
        method="L-BFGS-B",
        bounds=(exchange_bounds_pJ_per_m, demagnetizing_bounds_mT),
    )
    if optimum.success and np.isfinite(optimum.fun):
        best_a, best_hd = (float(value) for value in optimum.x)
    else:
        # The explicitly evaluated grid is a valid, reproducible fallback if
        # the local two-parameter refinement cannot make progress.
        best_a = float(a_values[column])
        best_hd = float(hd_values[row])
    best_fit = fit_at_exchange_stiffness(
        series,
        best_a,
        geometry,
        q_range=q_range,
        minimum_applied_field_mT=minimum_applied_field_mT,
        demagnetizing_field_override_mT=best_hd,
    )
    global_degrees = max(best_fit.degrees_of_freedom - 2, 0)
    best_fit = replace(
        best_fit,
        degrees_of_freedom=global_degrees,
        reduced_chi_square=(
            best_fit.chi_square / global_degrees if global_degrees > 0 else np.nan
        ),
    )
    return DemagnetizingFitResult(best_fit, best_hd, a_values, hd_values, surface)


def _fit_prepared(
    data: PreparedSeries,
    geometry: ExperimentGeometry,
    exchange: float,
    response_h: NDArray[np.float64],
    response_m: NDArray[np.float64] | None,
) -> FixedExchangeFit:
    n_fields, n_q = data.intensity.shape
    n_parameters = 3 if response_m is not None else 2
    coefficients = np.zeros((n_parameters, n_q))
    standard_errors = np.full_like(coefficients, np.nan)
    fitted = np.zeros_like(data.intensity)
    condition = np.zeros(n_q)
    rank = np.zeros(n_q, dtype=int)
    active = np.zeros((n_parameters, n_q), dtype=bool)
    effective_parameters = 0

    for q_index in range(n_q):
        columns = [np.ones(n_fields), response_h[:, q_index]]
        if response_m is not None:
            columns.append(response_m[:, q_index])
        design = np.column_stack(columns)
        sigma = data.uncertainty[:, q_index]
        whitened_design = design / sigma[:, None]
        whitened_intensity = data.intensity[:, q_index] / sigma
        solution, _ = nnls(whitened_design, whitened_intensity)
        coefficients[:, q_index] = solution
        fitted[:, q_index] = design @ solution
        rank[q_index] = np.linalg.matrix_rank(whitened_design)
        condition[q_index] = np.linalg.cond(whitened_design)

        tolerance = 1.0e-10 * max(1.0, float(np.max(solution)))
        free = solution > tolerance
        active[:, q_index] = ~free
        effective_parameters += int(np.count_nonzero(free))
        free_design = whitened_design[:, free]
        if free_design.size and np.linalg.matrix_rank(free_design) == free_design.shape[1]:
            # Work from the SVD pseudoinverse of the whitened design matrix.
            # This is algebraically equivalent to (X^T X)^-1 for a full-rank
            # unconstrained problem, but does not explicitly square cond(X).
            pseudoinverse = np.linalg.pinv(free_design)
            covariance = pseudoinverse @ pseudoinverse.T
            standard_errors[free, q_index] = np.sqrt(
                np.clip(np.diag(covariance), 0.0, None)
            )

    residuals = data.intensity - fitted
    standardized = residuals / data.uncertainty
    chi_square = float(np.sum(standardized**2))
    observations = int(data.intensity.size)
    degrees = max(observations - effective_parameters, 0)
    return FixedExchangeFit(
        geometry=geometry,
        exchange_stiffness_pJ_per_m=exchange,
        q=data.q,
        applied_fields_mT=data.applied_fields_mT,
        coefficients=coefficients,
        conditional_standard_errors=standard_errors,
        response_h=response_h,
        response_m=response_m,
        fitted_intensity=fitted,
        residuals=residuals,
        standardized_residuals=standardized,
        chi_square=chi_square,
        mean_chi_square=chi_square / observations,
        degrees_of_freedom=degrees,
        reduced_chi_square=chi_square / degrees if degrees > 0 else np.nan,
        condition_number=condition,
        rank=rank,
        active_constraints=active,
    )


def _insert_profile_point(a, chi, best_a, best_chi):
    position = int(np.searchsorted(a, best_a))
    return np.insert(a, position, best_a), np.insert(chi, position, best_chi)


def _profile_interval(a, chi, best_a, threshold, level):
    delta = chi - np.nanmin(chi)
    minimum = int(np.argmin(np.abs(a - best_a)))
    lower = _nearest_crossing(
        a[: minimum + 1][::-1], delta[: minimum + 1][::-1], threshold
    )
    upper = _nearest_crossing(a[minimum:], delta[minimum:], threshold)
    return ProfileInterval(level, threshold, lower, upper)


def _nearest_crossing(x, y, threshold):
    for index in range(1, x.size):
        if y[index] >= threshold and y[index - 1] < threshold:
            fraction = (threshold - y[index - 1]) / (y[index] - y[index - 1])
            return float(x[index - 1] + fraction * (x[index] - x[index - 1]))
    return None


def _curvature_standard_error(objective, optimum, lower, upper):
    step = max(abs(optimum) * 1.0e-4, (upper - lower) * 1.0e-6)
    if optimum - 2.0 * step <= lower or optimum + 2.0 * step >= upper:
        return None
    values = [
        objective(optimum + multiple * step) for multiple in (-2, -1, 0, 1, 2)
    ]
    second = (
        -values[0] + 16 * values[1] - 30 * values[2] + 16 * values[3] - values[4]
    ) / (12 * step**2)
    if not np.isfinite(second) or second <= 0.0:
        return None
    return float(np.sqrt(2.0 / second))
