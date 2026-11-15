"""Compact Matplotlib figures shared by scripts and the web application."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .data import FieldSeries
from .fit import DemagnetizingFitResult, ExchangeFitResult, StabilityMap


def plot_input_series(series: FieldSeries):
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for curve in series.curves:
        axis.errorbar(
            curve.q,
            curve.intensity,
            yerr=curve.uncertainty,
            marker="o",
            markersize=2.5,
            linewidth=0.9,
            capsize=1.5,
            label=f"{curve.applied_field_mT:g} mT",
        )
    axis.set_xscale("log")
    if np.all(series.intensity > 0.0):
        axis.set_yscale("log")
    axis.set_xlabel(r"$q$ (nm$^{-1}$)")
    axis.set_ylabel(r"$I(q)$")
    axis.set_title("Field-dependent input data", loc="left", fontweight="semibold")
    axis.legend(ncol=2, fontsize=8, frameon=False)
    axis.grid(alpha=0.18)
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.90)
    return figure


def plot_exchange_profile(result: ExchangeFitResult):
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    delta = result.profile_chi_square - np.min(result.profile_chi_square)
    axis.plot(result.profile_exchange_stiffness_pJ_per_m, delta, color="#136f7a")
    best = result.best_fit.exchange_stiffness_pJ_per_m
    axis.axvline(best, color="#b9632c", linestyle="--", label=f"best A = {best:.4g} pJ/m")
    axis.axhline(1.0, color="#667085", linestyle=":", label=r"$\Delta\chi^2=1$")
    axis.axhline(3.8414588, color="#98a2b3", linestyle="--", label=r"$\Delta\chi^2=3.84$")
    axis.set_xlabel(r"exchange stiffness $A$ (pJ m$^{-1}$)")
    axis.set_ylabel(r"$\chi^2(A)-\chi^2_{\min}$")
    axis.set_title("Exchange-stiffness profile", loc="left", fontweight="semibold")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.18)
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.90)
    return figure


def plot_scattering_components(result: ExchangeFitResult):
    fit = result.best_fit
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    colors = ("#172033", "#136f7a", "#b9632c")
    for index, (name, color) in enumerate(zip(fit.parameter_names, colors)):
        values = fit.coefficients[index]
        errors = fit.conditional_standard_errors[index]
        axis.plot(fit.q, values, color=color, label=name)
        finite = np.isfinite(errors)
        if np.any(finite):
            axis.fill_between(
                fit.q[finite],
                np.maximum(values[finite] - errors[finite], np.finfo(float).tiny),
                values[finite] + errors[finite],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
    axis.set_xscale("log")
    if np.all(fit.coefficients > 0.0):
        axis.set_yscale("log")
    axis.set_xlabel(r"$q$ (nm$^{-1}$)")
    axis.set_ylabel("scattering function")
    axis.set_title("Recovered q-dependent components", loc="left", fontweight="semibold")
    axis.legend(frameon=False)
    axis.grid(alpha=0.18)
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.90)
    return figure


def plot_reconstruction(result: ExchangeFitResult):
    fit = result.best_fit
    observed = fit.fitted_intensity + fit.residuals
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.9, observed.shape[0]))
    for index, field in enumerate(fit.applied_fields_mT):
        axis.plot(fit.q, observed[index], "o", color=colors[index], markersize=2.5)
        axis.plot(fit.q, fit.fitted_intensity[index], color=colors[index], linewidth=1.2, label=f"{field:g} mT")
    axis.set_xscale("log")
    if np.all(observed > 0.0):
        axis.set_yscale("log")
    axis.set_xlabel(r"$q$ (nm$^{-1}$)")
    axis.set_ylabel(r"$I(q)$")
    axis.set_title("Experimental data and reconstruction", loc="left", fontweight="semibold")
    axis.legend(ncol=2, fontsize=8, frameon=False)
    axis.grid(alpha=0.18)
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.90)
    return figure


def plot_fit_diagnostics(result: ExchangeFitResult):
    fit = result.best_fit
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].plot(fit.q, fit.condition_number, color="#4657a8")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$q$ (nm$^{-1}$)")
    axes[0].set_ylabel("condition number")
    axes[0].set_title("Whitened response matrix", loc="left", fontweight="semibold")
    axes[0].grid(alpha=0.18)

    image = axes[1].pcolormesh(
        fit.q,
        fit.applied_fields_mT,
        fit.standardized_residuals,
        shading="auto",
        cmap="coolwarm",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$q$ (nm$^{-1}$)")
    axes[1].set_ylabel(r"$\mu_0 H_0$ (mT)")
    axes[1].set_title("Standardized residuals", loc="left", fontweight="semibold")
    colorbar = figure.colorbar(image, ax=axes[1], fraction=0.048, pad=0.04)
    colorbar.set_label(r"$(I_{exp}-I_{fit})/\sigma$")
    figure.subplots_adjust(left=0.08, right=0.94, bottom=0.15, top=0.88, wspace=0.30)
    return figure


def plot_stability_map(result: StabilityMap):
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    image = axis.imshow(
        result.exchange_stiffness_pJ_per_m,
        origin="lower",
        aspect="auto",
        extent=(
            result.q_max_values[0],
            result.q_max_values[-1],
            result.minimum_field_values_mT[0],
            result.minimum_field_values_mT[-1],
        ),
        cmap="viridis",
    )
    axis.set_xlabel(r"$q_{max}$ (nm$^{-1}$)")
    axis.set_ylabel(r"minimum $\mu_0H_0$ (mT)")
    axis.set_title(r"Stability of $A_{best}$", loc="left", fontweight="semibold")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label(r"$A_{best}$ (pJ m$^{-1}$)")
    figure.subplots_adjust(left=0.13, right=0.88, bottom=0.14, top=0.90)
    return figure


def plot_demagnetizing_surface(result: DemagnetizingFitResult):
    """Plot the explicitly sampled joint A/Hd objective surface."""
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    delta = result.chi_square_surface - np.nanmin(result.chi_square_surface)
    image = axis.imshow(
        np.log10(1.0 + np.maximum(delta, 0.0)),
        origin="lower",
        aspect="auto",
        extent=(
            result.exchange_values_pJ_per_m[0],
            result.exchange_values_pJ_per_m[-1],
            result.demagnetizing_values_mT[0],
            result.demagnetizing_values_mT[-1],
        ),
        cmap="viridis",
    )
    axis.plot(
        result.best_fit.exchange_stiffness_pJ_per_m,
        result.best_demagnetizing_field_mT,
        marker="x",
        markersize=10,
        markeredgewidth=2,
        color="white",
        label="refined optimum",
    )
    axis.set_xlabel(r"exchange stiffness $A$ (pJ m$^{-1}$)")
    axis.set_ylabel(r"common $\mu_0H_d$ (mT)")
    axis.set_title(r"Joint $A/H_d$ objective surface", loc="left", fontweight="semibold")
    axis.legend(frameon=False, labelcolor="white", loc="upper right")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label(r"$\log_{10}(1+\chi^2-\chi^2_{min})$")
    figure.subplots_adjust(left=0.13, right=0.88, bottom=0.14, top=0.90)
    return figure
