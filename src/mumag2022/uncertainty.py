"""Simulation-based checks for exchange-stiffness uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import ExperimentGeometry, FieldSeries, ScatteringCurve, prepare_series
from .fit import ExchangeFitResult, fit_exchange_stiffness


@dataclass(frozen=True)
class BootstrapResult:
    """Parametric-bootstrap distribution conditional on the fitted model."""

    exchange_samples_pJ_per_m: NDArray[np.float64]
    median_pJ_per_m: float
    interval_68_pJ_per_m: tuple[float, float]
    interval_95_pJ_per_m: tuple[float, float]
    lower_boundary_fraction: float
    upper_boundary_fraction: float


def parametric_bootstrap(
    series: FieldSeries,
    fit: ExchangeFitResult,
    geometry: ExperimentGeometry | str,
    *,
    replicates: int = 200,
    seed: int = 7,
    profile_samples: int = 41,
) -> BootstrapResult:
    """Refit Gaussian draws using the experimental pixel uncertainties.

    This assesses the estimator under the fitted MuMag model and supplied
    independent Gaussian errors. It does not validate either assumption.
    """
    if replicates < 10:
        raise ValueError("at least ten bootstrap replicates are required")
    geometry = ExperimentGeometry(geometry)
    selected = prepare_series(
        series,
        q_range=fit.q_range,
        minimum_applied_field_mT=fit.minimum_applied_field_mT,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(int(replicates))
    for position in range(int(replicates)):
        simulated = fit.best_fit.fitted_intensity + rng.normal(
            0.0, selected.uncertainty
        )
        curves = tuple(
            ScatteringCurve(
                selected.q,
                simulated[index],
                selected.uncertainty[index],
                selected.applied_fields_mT[index],
                selected.saturation_magnetizations_mT[index],
                selected.demagnetizing_fields_mT[index],
                label=f"bootstrap_{position}_{index}",
            )
            for index in range(selected.applied_fields_mT.size)
        )
        result = fit_exchange_stiffness(
            FieldSeries(curves),
            geometry,
            exchange_bounds_pJ_per_m=fit.exchange_bounds_pJ_per_m,
            profile_samples=profile_samples,
            q_range=fit.q_range,
            minimum_applied_field_mT=fit.minimum_applied_field_mT,
        )
        samples[position] = result.best_fit.exchange_stiffness_pJ_per_m

    lower, upper = fit.exchange_bounds_pJ_per_m
    tolerance = 1.0e-8 * (upper - lower)
    return BootstrapResult(
        exchange_samples_pJ_per_m=samples,
        median_pJ_per_m=float(np.median(samples)),
        interval_68_pJ_per_m=tuple(np.quantile(samples, [0.16, 0.84])),
        interval_95_pJ_per_m=tuple(np.quantile(samples, [0.025, 0.975])),
        lower_boundary_fraction=float(np.mean(samples <= lower + tolerance)),
        upper_boundary_fraction=float(np.mean(samples >= upper - tolerance)),
    )
