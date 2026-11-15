"""Micromagnetic response functions used by MuMag2022."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import ExperimentGeometry


MU_0 = 4.0 * np.pi * 1.0e-7


def response_functions(
    q_nm_inv: ArrayLike,
    applied_field_mT: ArrayLike,
    saturation_magnetization_mT: ArrayLike,
    demagnetizing_field_mT: ArrayLike,
    exchange_stiffness_pJ_per_m: float,
    geometry: ExperimentGeometry | str,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Return azimuthally averaged ``R_H`` and, if applicable, ``R_M``."""
    geometry = ExperimentGeometry(geometry)
    q = np.asarray(q_nm_inv, dtype=float)
    h0 = np.asarray(applied_field_mT, dtype=float)
    ms = np.asarray(saturation_magnetization_mT, dtype=float)
    hd = np.asarray(demagnetizing_field_mT, dtype=float)
    if q.ndim != 1 or h0.ndim != 1 or ms.ndim != 1 or hd.ndim != 1:
        raise ValueError("q and magnetic metadata must be one-dimensional")
    if not (h0.shape == ms.shape == hd.shape):
        raise ValueError("field and magnetization metadata must have equal lengths")
    exchange = float(exchange_stiffness_pJ_per_m)
    if not np.isfinite(exchange) or exchange <= 0.0:
        raise ValueError("exchange stiffness must be finite and positive")

    internal_mT = h0 - hd
    if np.any(internal_mT <= 0.0):
        raise ValueError("all internal fields H0 - Hd must be positive")
    internal = internal_mT * 1.0e-3 / MU_0
    magnetization = ms * 1.0e-3 / MU_0
    q_si = q * 1.0e9
    exchange_si = exchange * 1.0e-12

    effective = internal[:, None] + (
        2.0 * exchange_si * q_si[None, :] ** 2
        / (MU_0 * magnetization[:, None])
    )
    p = magnetization[:, None] / effective

    if geometry is ExperimentGeometry.PARALLEL:
        return 0.5 * p**2, None

    root = np.sqrt(1.0 + p)
    response_h = 0.25 * p**2 * (2.0 + 1.0 / root)
    response_m = 0.5 * (root - 1.0)
    return response_h, response_m


def characteristic_q_max_nm_inv(
    exchange_stiffness_pJ_per_m: float,
    saturation_magnetization_mT: float,
    maximum_field_mT: float,
) -> float:
    """Return the paper's characteristic q_max estimate in nm^-1."""
    exchange = float(exchange_stiffness_pJ_per_m) * 1.0e-12
    magnetization = float(saturation_magnetization_mT) * 1.0e-3 / MU_0
    field = float(maximum_field_mT) * 1.0e-3 / MU_0
    if exchange <= 0.0 or magnetization <= 0.0 or field <= 0.0:
        raise ValueError("A, M_s, and H_max must be positive")
    return float(np.sqrt(MU_0 * magnetization * field / (2.0 * exchange)) / 1.0e9)
