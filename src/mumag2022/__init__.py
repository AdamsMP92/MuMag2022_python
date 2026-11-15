"""Public API for MuMag2022 Python."""

from .data import ExperimentGeometry, FieldSeries, PreparedSeries, ScatteringCurve, prepare_series
from .fit import (
    DemagnetizingFitResult,
    ExchangeFitResult,
    FixedExchangeFit,
    ProfileInterval,
    StabilityMap,
    fit_at_exchange_stiffness,
    fit_exchange_and_demagnetizing_field,
    fit_exchange_stiffness,
    stability_map,
)
from .io import load_legacy_directory, load_legacy_payloads, parse_legacy_filename
from .response import MU_0, characteristic_q_max_nm_inv, response_functions
from .synthetic import generate_synthetic_series
from .uncertainty import BootstrapResult, parametric_bootstrap
from .web_support import build_result_archive

__all__ = [
    "DemagnetizingFitResult",
    "BootstrapResult",
    "ExchangeFitResult",
    "ExperimentGeometry",
    "FieldSeries",
    "FixedExchangeFit",
    "MU_0",
    "PreparedSeries",
    "ProfileInterval",
    "ScatteringCurve",
    "StabilityMap",
    "characteristic_q_max_nm_inv",
    "build_result_archive",
    "fit_at_exchange_stiffness",
    "fit_exchange_and_demagnetizing_field",
    "fit_exchange_stiffness",
    "generate_synthetic_series",
    "load_legacy_directory",
    "load_legacy_payloads",
    "parse_legacy_filename",
    "parametric_bootstrap",
    "prepare_series",
    "response_functions",
    "stability_map",
]
