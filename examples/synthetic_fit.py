"""Minimal reproducible MuMag2022 fit using synthetic data."""

import os
from pathlib import Path

cache_directory = Path(".cache/matplotlib").resolve()
cache_directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))

import matplotlib.pyplot as plt

from mumag2022 import ExperimentGeometry, fit_exchange_stiffness, generate_synthetic_series
from mumag2022.plotting import plot_exchange_profile, plot_fit_diagnostics


output_directory = Path("examples/output")
output_directory.mkdir(parents=True, exist_ok=True)

series = generate_synthetic_series(
    geometry=ExperimentGeometry.PERPENDICULAR,
    exchange_stiffness_pJ_per_m=6.5,
    relative_noise=0.02,
    seed=7,
)

result = fit_exchange_stiffness(
    series,
    ExperimentGeometry.PERPENDICULAR,
    exchange_bounds_pJ_per_m=(0.5, 20.0),
    profile_samples=61,
)

print(f"best A = {result.best_fit.exchange_stiffness_pJ_per_m:.4g} pJ/m")
print(f"reduced chi-square = {result.best_fit.reduced_chi_square:.4g}")
print(f"maximum condition number = {result.best_fit.condition_number.max():.4g}")

profile = plot_exchange_profile(result)
profile.savefig(output_directory / "synthetic_exchange_profile.png", dpi=180)
plt.close(profile)

diagnostics = plot_fit_diagnostics(result)
diagnostics.savefig(output_directory / "synthetic_diagnostics.png", dpi=180)
plt.close(diagnostics)

