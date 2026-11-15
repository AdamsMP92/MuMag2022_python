"""Synthetic field-series generators for verification and demonstration."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .data import ExperimentGeometry, FieldSeries, ScatteringCurve
from .response import response_functions


def generate_synthetic_series(
    *,
    geometry: ExperimentGeometry | str = ExperimentGeometry.PERPENDICULAR,
    q: ArrayLike | None = None,
    applied_fields_mT: ArrayLike = (60.0, 100.0, 200.0, 400.0, 800.0, 1200.0),
    saturation_magnetization_mT: float = 1340.0,
    demagnetizing_field_mT: float = 20.0,
    exchange_stiffness_pJ_per_m: float = 6.5,
    relative_noise: float = 0.02,
    seed: int = 7,
) -> FieldSeries:
    """Generate a reproducible Lorentzian-spectrum MuMag test series."""
    geometry = ExperimentGeometry(geometry)
    if q is None:
        q = np.geomspace(0.025, 0.35, 55)
    q = np.asarray(q, dtype=float)
    fields = np.asarray(applied_fields_mT, dtype=float)
    magnetization = np.full(fields.size, float(saturation_magnetization_mT))
    demag = np.full(fields.size, float(demagnetizing_field_mT))
    response_h, response_m = response_functions(
        q,
        fields,
        magnetization,
        demag,
        exchange_stiffness_pJ_per_m,
        geometry,
    )
    residual = 180.0 / (1.0 + (q * 24.0) ** 2) ** 2 + 2.0
    scattering_h = 680.0 / (1.0 + (q * 7.0) ** 2) ** 2
    model = residual[None, :] + response_h * scattering_h[None, :]
    if response_m is not None:
        scattering_m = 420.0 / (1.0 + (q * 10.0) ** 2) ** 2
        model = model + response_m * scattering_m[None, :]
    scale = max(float(np.max(model)), 1.0)
    uncertainty = relative_noise * np.maximum(np.abs(model), 0.01 * scale)
    rng = np.random.default_rng(seed)
    noisy = model + rng.normal(0.0, uncertainty)
    curves = tuple(
        ScatteringCurve(
            q,
            noisy[index],
            uncertainty[index],
            fields[index],
            magnetization[index],
            demag[index],
            label=f"synthetic_{fields[index]:g}_mT",
        )
        for index in range(fields.size)
    )
    return FieldSeries(curves)
