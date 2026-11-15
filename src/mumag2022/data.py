"""Validated data structures for field-dependent one-dimensional SANS data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ExperimentGeometry(str, Enum):
    """Orientation of applied field relative to the incident neutron beam."""

    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"


@dataclass(frozen=True)
class ScatteringCurve:
    """One azimuthally averaged SANS curve and its magnetic metadata.

    The magnetic metadata are ``mu_0``-scaled values in mT, matching the
    original MuMag2022 input convention.
    """

    q: ArrayLike
    intensity: ArrayLike
    uncertainty: ArrayLike
    applied_field_mT: float
    saturation_magnetization_mT: float
    demagnetizing_field_mT: float
    label: str = ""

    def __post_init__(self) -> None:
        q = _one_dimensional(self.q, "q")
        intensity = _one_dimensional(self.intensity, "intensity")
        uncertainty = _one_dimensional(self.uncertainty, "uncertainty")
        if not (q.size == intensity.size == uncertainty.size):
            raise ValueError("q, intensity, and uncertainty must have equal lengths")
        if q.size < 2:
            raise ValueError("each scattering curve needs at least two q values")
        if np.any(~np.isfinite(q)) or np.any(q <= 0.0):
            raise ValueError("q values must be finite and positive")
        if np.any(np.diff(q) <= 0.0):
            raise ValueError("q values must be strictly increasing")
        if np.any(~np.isfinite(intensity)):
            raise ValueError("intensities must be finite")
        if np.any(~np.isfinite(uncertainty)) or np.any(uncertainty <= 0.0):
            raise ValueError("uncertainties must be finite and strictly positive")

        metadata = np.asarray(
            [
                self.applied_field_mT,
                self.saturation_magnetization_mT,
                self.demagnetizing_field_mT,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(metadata)):
            raise ValueError("magnetic metadata must be finite")
        if self.saturation_magnetization_mT <= 0.0:
            raise ValueError("saturation magnetization must be positive")
        if self.internal_field_mT <= 0.0:
            raise ValueError("the internal field H0 - Hd must be positive")

        object.__setattr__(self, "q", q)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "uncertainty", uncertainty)

    @property
    def internal_field_mT(self) -> float:
        return float(self.applied_field_mT - self.demagnetizing_field_mT)


@dataclass(frozen=True)
class FieldSeries:
    """A field-dependent series defined on a common q grid."""

    curves: tuple[ScatteringCurve, ...]

    def __post_init__(self) -> None:
        curves = tuple(self.curves)
        if len(curves) < 2:
            raise ValueError("a field series requires at least two curves")
        reference = np.asarray(curves[0].q)
        for curve in curves[1:]:
            if curve.q.shape != reference.shape or not np.allclose(
                curve.q, reference, rtol=1.0e-9, atol=1.0e-12
            ):
                raise ValueError(
                    "all curves must share the same q grid; resampling is never implicit"
                )
        fields = np.asarray([curve.applied_field_mT for curve in curves])
        if np.unique(fields).size != fields.size:
            raise ValueError("applied field values must be unique")
        object.__setattr__(
            self,
            "curves",
            tuple(sorted(curves, key=lambda curve: curve.applied_field_mT)),
        )

    @property
    def q(self) -> NDArray[np.float64]:
        return np.asarray(self.curves[0].q, dtype=float)

    @property
    def applied_fields_mT(self) -> NDArray[np.float64]:
        return np.asarray([curve.applied_field_mT for curve in self.curves])

    @property
    def saturation_magnetizations_mT(self) -> NDArray[np.float64]:
        return np.asarray(
            [curve.saturation_magnetization_mT for curve in self.curves]
        )

    @property
    def demagnetizing_fields_mT(self) -> NDArray[np.float64]:
        return np.asarray([curve.demagnetizing_field_mT for curve in self.curves])

    @property
    def intensity(self) -> NDArray[np.float64]:
        return np.vstack([curve.intensity for curve in self.curves])

    @property
    def uncertainty(self) -> NDArray[np.float64]:
        return np.vstack([curve.uncertainty for curve in self.curves])


@dataclass(frozen=True)
class PreparedSeries:
    """Numerical arrays after explicit q and field selection."""

    q: NDArray[np.float64]
    intensity: NDArray[np.float64]
    uncertainty: NDArray[np.float64]
    applied_fields_mT: NDArray[np.float64]
    saturation_magnetizations_mT: NDArray[np.float64]
    demagnetizing_fields_mT: NDArray[np.float64]
    labels: tuple[str, ...]


def prepare_series(
    series: FieldSeries,
    *,
    q_range: tuple[float, float] | None = None,
    minimum_applied_field_mT: float | None = None,
    demagnetizing_field_override_mT: float | None = None,
) -> PreparedSeries:
    """Select data without silently changing q grids or metadata."""
    field_mask = np.ones(len(series.curves), dtype=bool)
    if minimum_applied_field_mT is not None:
        field_mask &= series.applied_fields_mT >= float(minimum_applied_field_mT)
    indices = np.flatnonzero(field_mask)
    if indices.size < 2:
        raise ValueError("the selected field range contains fewer than two curves")

    q = series.q
    q_mask = np.ones(q.size, dtype=bool)
    if q_range is not None:
        lower, upper = (float(value) for value in q_range)
        if not 0.0 < lower < upper:
            raise ValueError("q_range must satisfy 0 < q_min < q_max")
        q_mask &= (q >= lower) & (q <= upper)
    if np.count_nonzero(q_mask) < 2:
        raise ValueError("the selected q range contains fewer than two points")

    demag = series.demagnetizing_fields_mT[indices]
    if demagnetizing_field_override_mT is not None:
        demag = np.full(indices.size, float(demagnetizing_field_override_mT))
    applied = series.applied_fields_mT[indices]
    if np.any(applied - demag <= 0.0):
        raise ValueError("the selected internal fields H0 - Hd must all be positive")

    return PreparedSeries(
        q=np.asarray(q[q_mask]),
        intensity=np.asarray(series.intensity[np.ix_(indices, q_mask)]),
        uncertainty=np.asarray(series.uncertainty[np.ix_(indices, q_mask)]),
        applied_fields_mT=np.asarray(applied),
        saturation_magnetizations_mT=np.asarray(
            series.saturation_magnetizations_mT[indices]
        ),
        demagnetizing_fields_mT=np.asarray(demag),
        labels=tuple(series.curves[index].label for index in indices),
    )


def _one_dimensional(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return np.array(array, copy=True)
