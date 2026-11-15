"""In-memory result serialization for browser and script frontends."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping

import numpy as np

from .fit import ExchangeFitResult
from .io import rows_to_csv


def build_result_archive(
    result: ExchangeFitResult,
    *,
    configuration: Mapping[str, object],
    figures: Mapping[str, bytes] | None = None,
) -> bytes:
    """Return a reproducible ZIP archive with fit data and diagnostics."""
    fit = result.best_fit
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("configuration.json", json.dumps(configuration, indent=2, sort_keys=True))
        archive.writestr("fit_summary.json", json.dumps(_summary(result), indent=2, sort_keys=True))
        archive.writestr(
            "exchange_profile.csv",
            rows_to_csv(
                ("A_pJ_per_m", "chi_square", "delta_chi_square"),
                zip(
                    result.profile_exchange_stiffness_pJ_per_m,
                    result.profile_chi_square,
                    result.profile_chi_square - np.min(result.profile_chi_square),
                ),
            ),
        )
        component_columns = [fit.q]
        headers = ["q_nm_inv"]
        for index, name in enumerate(fit.parameter_names):
            headers.extend((name, f"{name}_conditional_standard_error", f"{name}_at_zero_boundary"))
            component_columns.extend(
                (
                    fit.coefficients[index],
                    fit.conditional_standard_errors[index],
                    fit.active_constraints[index].astype(int),
                )
            )
        archive.writestr("scattering_components.csv", rows_to_csv(headers, zip(*component_columns)))
        rows = []
        observed = fit.fitted_intensity + fit.residuals
        for field_index, field in enumerate(fit.applied_fields_mT):
            rows.extend(
                zip(
                    np.full(fit.q.size, field),
                    fit.q,
                    observed[field_index],
                    fit.fitted_intensity[field_index],
                    fit.residuals[field_index],
                    fit.standardized_residuals[field_index],
                )
            )
        archive.writestr(
            "reconstructed_curves.csv",
            rows_to_csv(
                ("applied_field_mT", "q_nm_inv", "observed", "fitted", "residual", "standardized_residual"),
                rows,
            ),
        )
        archive.writestr(
            "conditioning.csv",
            rows_to_csv(("q_nm_inv", "condition_number", "rank"), zip(fit.q, fit.condition_number, fit.rank)),
        )
        for filename, payload in (figures or {}).items():
            archive.writestr(filename, payload)
    return buffer.getvalue()


def _summary(result: ExchangeFitResult) -> dict[str, object]:
    fit = result.best_fit
    return {
        "geometry": fit.geometry.value,
        "exchange_stiffness_pJ_per_m": fit.exchange_stiffness_pJ_per_m,
        "curvature_standard_error_pJ_per_m": result.curvature_standard_error_pJ_per_m,
        "profile_interval_68_pJ_per_m": [result.interval_68.lower, result.interval_68.upper],
        "profile_interval_95_pJ_per_m": [result.interval_95.lower, result.interval_95.upper],
        "chi_square": fit.chi_square,
        "mean_chi_square": fit.mean_chi_square,
        "reduced_chi_square": fit.reduced_chi_square,
        "degrees_of_freedom": fit.degrees_of_freedom,
        "characteristic_q_max_nm_inv": result.characteristic_q_max_nm_inv,
        "maximum_condition_number": float(np.max(fit.condition_number)),
        "fraction_active_constraints": float(np.mean(fit.active_constraints)),
    }
