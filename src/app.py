"""Browser-native Shiny interface for MuMag2022 Python."""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
matplotlib_cache = Path(tempfile.gettempdir()) / "mumag2022-matplotlib"
matplotlib_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

from mumag2022 import (
    DemagnetizingFitResult,
    ExchangeFitResult,
    ExperimentGeometry,
    FieldSeries,
    StabilityMap,
    build_result_archive,
    fit_exchange_and_demagnetizing_field,
    fit_exchange_stiffness,
    generate_synthetic_series,
    load_legacy_payloads,
    stability_map,
)
from mumag2022.plotting import (
    plot_demagnetizing_surface,
    plot_exchange_profile,
    plot_fit_diagnostics,
    plot_input_series,
    plot_reconstruction,
    plot_scattering_components,
    plot_stability_map,
)

@dataclass(frozen=True)
class AnalysisState:
    result: ExchangeFitResult
    archive: bytes
    configuration: dict[str, object]


def _section(index: str, title: str):
    return ui.div(
        ui.tags.span(index, class_="section-index"),
        ui.tags.span(title, class_="section-label"),
        class_="input-section-title",
    )


def _note(text: str):
    return ui.tags.small(text, class_="field-note")


def _metric(label: str, value: str, detail: str, accent: str = "teal"):
    return ui.div(
        ui.tags.span(label, class_="metric-label"),
        ui.div(value, class_="metric-value"),
        ui.tags.p(detail, class_="metric-detail"),
        class_=f"metric-card metric-{accent}",
    )


def _figure_bytes(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    return buffer.getvalue()


input_panel = ui.div(
    ui.tags.span("ANALYSIS INPUT", class_="eyebrow"),
    ui.h2("Configure the field-series fit"),
    ui.tags.p(
        "Load a legacy MuMag field series or use the reproducible synthetic example.",
        class_="panel-intro",
    ),
    _section("01", "Data source"),
    ui.input_select(
        "data_source",
        "Input data",
        {"demo": "Synthetic verification series", "upload": "Upload legacy files"},
        selected="demo",
    ),
    ui.panel_conditional(
        "input.data_source === 'upload'",
        ui.input_file(
            "dataset",
            "Field-dependent curves",
            accept=[".csv", ".dat", ".txt"],
            multiple=True,
            button_label="Browse…",
            placeholder="Select all field curves",
        ),
        _note("Headerless q, I, sigma files named index_H0_Ms_Hd.csv"),
    ),
    _section("02", "Micromagnetic model"),
    ui.input_select(
        "geometry",
        "Scattering geometry",
        {
            "perpendicular": "Field perpendicular to beam",
            "parallel": "Field parallel to beam",
        },
        selected="perpendicular",
    ),
    ui.div(
        ui.input_numeric("q_min", "q min (nm⁻¹)", 0.025, min=1.0e-9, step=0.005),
        ui.input_numeric("q_max", "q max (nm⁻¹)", 0.35, min=1.0e-9, step=0.01),
        class_="input-pair",
    ),
    ui.input_numeric("field_min", "Minimum applied field (mT)", 60.0, min=0.0, step=10.0),
    _section("03", "Exchange profile"),
    ui.div(
        ui.input_numeric("a_min", "A min (pJ/m)", 0.5, min=1.0e-6, step=0.5),
        ui.input_numeric("a_max", "A max (pJ/m)", 20.0, min=1.0e-6, step=0.5),
        class_="input-pair",
    ),
    ui.input_slider("profile_samples", "Profile samples", 21, 161, 61, step=10),
    _note("A local bounded refinement follows the global profile scan."),
    ui.input_action_button("run_analysis", "Run MuMag analysis", class_="primary-action"),
    ui.output_ui("analysis_message"),
    class_="input-panel",
)


analyzer_panel = ui.div(
    ui.div(
        input_panel,
        ui.div(
            ui.div(
                ui.tags.span("DATA CHECK", class_="eyebrow"),
                ui.h2("Field series"),
                ui.output_ui("metadata_table"),
                class_="card-heading",
            ),
            ui.output_plot("input_plot", height="430px"),
            class_="content-card preview-card",
        ),
        class_="analysis-grid",
    ),
    ui.div(
        ui.tags.span("FIT RESULT", class_="eyebrow"),
        ui.h2("Exchange stiffness and model diagnostics"),
        ui.output_ui("result_metrics"),
        ui.navset_card_underline(
            ui.nav_panel("A profile", ui.output_plot("profile_plot", height="480px")),
            ui.nav_panel("Components", ui.output_plot("components_plot", height="480px")),
            ui.nav_panel("Reconstruction", ui.output_plot("reconstruction_plot", height="480px")),
            ui.nav_panel("Diagnostics", ui.output_plot("diagnostics_plot", height="470px")),
        ),
        ui.download_button("download_result", "Download reproducible result ZIP", class_="secondary-action"),
        class_="content-card result-card",
    ),
    class_="analyzer-panel",
)


stability_panel = ui.div(
    ui.div(
        ui.tags.span("ROBUSTNESS CHECK", class_="eyebrow"),
        ui.h2("Dependence on q max and minimum field"),
        ui.tags.p(
            "The same fit is repeated over a compact selection grid. A stable plateau is more informative than a single optimum.",
            class_="panel-intro",
        ),
        ui.input_slider("stability_steps", "Grid points per direction", 3, 9, 6),
        ui.input_action_button("run_stability", "Calculate stability map", class_="primary-action"),
        ui.output_ui("stability_message"),
        class_="stability-controls",
    ),
    ui.div(ui.output_plot("stability_plot", height="580px"), class_="content-card"),
    class_="stability-grid",
)


demag_panel = ui.div(
    ui.div(
        ui.tags.span("ADVANCED IDENTIFIABILITY CHECK", class_="eyebrow"),
        ui.h2("Joint exchange and demagnetizing-field fit"),
        ui.tags.p(
            "Replace the file-specific demagnetizing fields by one fitted common value and inspect the sampled two-parameter objective surface.",
            class_="panel-intro",
        ),
        ui.div(
            ui.input_numeric("hd_min", "Hd min (mT)", 0.0, min=0.0, step=1.0),
            ui.input_numeric("hd_max", "Hd max (mT)", 50.0, min=0.0, step=1.0),
            class_="input-pair",
        ),
        ui.input_slider("demag_samples", "Grid points per direction", 7, 25, 11, step=2),
        ui.input_action_button("run_demag", "Calculate joint fit", class_="primary-action"),
        ui.output_ui("demag_message"),
        class_="stability-controls",
    ),
    ui.div(
        ui.output_ui("demag_metrics"),
        ui.output_plot("demag_plot", height="580px"),
        class_="content-card",
    ),
    class_="stability-grid",
)


how_to_panel = ui.tags.article(
    ui.tags.span("VERIFICATION & INTERPRETATION", class_="eyebrow"),
    ui.h2("A transparent MuMag workflow"),
    ui.div(
        *[
            ui.div(
                ui.tags.span(index, class_="workflow-index"),
                ui.div(ui.h3(title), ui.tags.p(text)),
                class_="workflow-step",
            )
            for index, title, text in (
                ("01", "Load the full field series", "Use several applied fields on one common q grid and inspect the parsed magnetic metadata."),
                ("02", "Select the valid regime", "Choose fields in the approach-to-saturation regime and a q interval that retains measurable field dependence."),
                ("03", "Profile A", "For every trial A, solve the non-negative linear response problem directly at each q."),
                ("04", "Inspect identifiability", "Check matrix rank, condition numbers, active zero boundaries, the profile shape, and standardized residuals."),
                ("05", "Test robustness", "Vary q max and the minimum field. Report a stable region rather than one isolated fit."),
                ("06", "Export the evidence", "Save configuration, profiles, components, reconstructions, and diagnostics together."),
            )
        ],
        class_="workflow-grid",
    ),
    ui.div(
        ui.h3("Scientific boundary"),
        ui.tags.p(
            "The calculation assumes unpolarized azimuthally averaged data, positive internal fields, a statistically isotropic material, one exchange stiffness, and field-independent q-dependent scattering functions over the selected range. A good numerical fit does not prove these assumptions."
        ),
        class_="guidance-card warning-card",
    ),
    ui.div(
        ui.h3("Uncertainty boundary"),
        ui.tags.p(
            "The displayed profile intervals use conventional one-parameter delta-chi-square thresholds. They are approximations when non-negative constraints are active, the minimum lies at a search boundary, errors are correlated, or the supplied uncertainties are not calibrated standard deviations."
        ),
        class_="guidance-card",
    ),
    class_="documentation-panel",
)


app_ui = ui.page_bootstrap(
    ui.head_content(
        ui.tags.meta(name="description", content="Browser-native MuMag2022 magnetic SANS analysis"),
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin="anonymous"),
        ui.tags.link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,600&display=swap",
        ),
    ),
    ui.include_css(APP_DIR / "www" / "styles.css"),
    ui.div(
        ui.tags.header(
            ui.div(
                ui.div(
                    ui.tags.span("MICHAEL P. ADAMS · MAGNETIC SANS", class_="hero-kicker"),
                    ui.h1("MuMag2022 Python"),
                    ui.tags.p("Exchange-stiffness analysis with explicit conditioning, profile, residual, and stability diagnostics."),
                ),
                ui.div(
                    ui.tags.span(class_="runtime-dot"),
                    ui.div(ui.tags.strong("Browser-native Python"), ui.tags.small("Shinylive · data stay local")),
                    class_="runtime-badge",
                ),
                class_="hero-inner",
            ),
            class_="app-hero",
        ),
        ui.div(
            ui.navset_card_underline(
                ui.nav_panel("Analyzer", analyzer_panel),
                ui.nav_panel("Stability", stability_panel),
                ui.nav_panel("Demag fit", demag_panel),
                ui.nav_panel("How to use", how_to_panel),
            ),
            class_="main-nav-wrap",
        ),
        ui.tags.footer(
            ui.tags.span("MuMag2022 Python · Michael P. Adams · "),
            ui.tags.a("Source code", href="https://github.com/AdamsMP92/MuMag2022_python", target="_blank"),
            class_="app-footer",
        ),
        class_="app-shell",
    ),
    title="MuMag2022 Python",
    lang="en",
)


def server(input: Inputs, output: Outputs, session: Session) -> None:
    analysis_state: reactive.Value[AnalysisState | None] = reactive.Value(None)
    analysis_error: reactive.Value[str | None] = reactive.Value(None)
    stability_state: reactive.Value[StabilityMap | None] = reactive.Value(None)
    stability_error: reactive.Value[str | None] = reactive.Value(None)
    demag_state: reactive.Value[DemagnetizingFitResult | None] = reactive.Value(None)
    demag_error: reactive.Value[str | None] = reactive.Value(None)

    @reactive.calc
    def field_series() -> FieldSeries:
        geometry = ExperimentGeometry(str(input.geometry()))
        if str(input.data_source()) == "demo":
            return generate_synthetic_series(geometry=geometry)
        records = input.dataset()
        req(records)
        payloads = {
            str(record["name"]): Path(str(record["datapath"])).read_bytes()
            for record in records
        }
        return load_legacy_payloads(payloads)

    @reactive.effect
    @reactive.event(input.data_source, input.dataset, input.geometry)
    def _synchronize_ranges() -> None:
        analysis_state.set(None)
        stability_state.set(None)
        demag_state.set(None)
        try:
            series = field_series()
        except Exception as error:
            analysis_error.set(str(error))
            return
        analysis_error.set(None)
        ui.update_numeric("q_min", value=float(series.q[0]), min=float(series.q[0]), max=float(series.q[-1]))
        ui.update_numeric("q_max", value=float(series.q[-1]), min=float(series.q[0]), max=float(series.q[-1]))
        ui.update_numeric("field_min", value=float(series.applied_fields_mT[0]), min=0.0, max=float(series.applied_fields_mT[-1]))

    def configuration() -> dict[str, object]:
        return {
            "geometry": str(input.geometry()),
            "data_source": str(input.data_source()),
            "q_range_nm_inv": [float(input.q_min()), float(input.q_max())],
            "minimum_applied_field_mT": float(input.field_min()),
            "exchange_bounds_pJ_per_m": [float(input.a_min()), float(input.a_max())],
            "profile_samples": int(input.profile_samples()),
        }

    @output
    @render.plot
    def input_plot():
        return plot_input_series(field_series())

    @output
    @render.ui
    def metadata_table():
        try:
            series = field_series()
        except Exception as error:
            return ui.div(str(error), class_="calculation-message message-error")
        rows = [
            ui.tags.tr(
                ui.tags.td(curve.label or f"curve {index + 1}"),
                ui.tags.td(f"{curve.applied_field_mT:g}"),
                ui.tags.td(f"{curve.saturation_magnetization_mT:g}"),
                ui.tags.td(f"{curve.demagnetizing_field_mT:g}"),
                ui.tags.td(f"{curve.internal_field_mT:g}"),
            )
            for index, curve in enumerate(series.curves)
        ]
        return ui.div(
            ui.tags.table(
                ui.tags.thead(ui.tags.tr(*(ui.tags.th(value) for value in ("curve", "H0 (mT)", "Ms (mT)", "Hd (mT)", "Hi (mT)")))),
                ui.tags.tbody(*rows),
                class_="result-table",
            ),
            class_="table-scroll",
        )

    @output
    @render.ui
    def analysis_message():
        error = analysis_error.get()
        if error:
            return ui.div(error, class_="calculation-message message-error")
        try:
            series = field_series()
        except Exception:
            return ui.div("Select all field curves to begin.", class_="calculation-message message-info")
        return ui.div(
            f"Ready: {len(series.curves)} fields and {series.q.size} common q points.",
            class_="calculation-message message-success",
        )

    @reactive.effect
    @reactive.event(input.run_analysis)
    def _run_analysis() -> None:
        try:
            config = configuration()
            with ui.Progress(min=0, max=1) as progress:
                progress.set(0.05, message="Profiling exchange stiffness")
                result = fit_exchange_stiffness(
                    field_series(),
                    ExperimentGeometry(config["geometry"]),
                    exchange_bounds_pJ_per_m=tuple(config["exchange_bounds_pJ_per_m"]),
                    profile_samples=int(config["profile_samples"]),
                    q_range=tuple(config["q_range_nm_inv"]),
                    minimum_applied_field_mT=float(config["minimum_applied_field_mT"]),
                )
                progress.set(0.82, message="Packaging diagnostics")
                figures = {
                    "exchange_profile.png": _figure_bytes(plot_exchange_profile(result)),
                    "components.png": _figure_bytes(plot_scattering_components(result)),
                    "reconstruction.png": _figure_bytes(plot_reconstruction(result)),
                    "diagnostics.png": _figure_bytes(plot_fit_diagnostics(result)),
                }
                archive = build_result_archive(result, configuration=config, figures=figures)
                analysis_state.set(AnalysisState(result, archive, config))
                analysis_error.set(None)
                progress.set(1.0, message="Analysis complete")
        except Exception as error:
            analysis_state.set(None)
            analysis_error.set(str(error))

    @output
    @render.ui
    def result_metrics():
        state = analysis_state.get()
        if state is None:
            return ui.tags.p("Run the analysis to display the result.", class_="empty-copy")
        result = state.result
        fit = result.best_fit
        sigma = result.curvature_standard_error_pJ_per_m
        sigma_text = "not available" if sigma is None else f"{sigma:.3g} pJ/m"
        boundary = float(np.mean(fit.active_constraints))
        return ui.div(
            _metric("Best A", f"{fit.exchange_stiffness_pJ_per_m:.4g} pJ/m", "profile minimum", "indigo"),
            _metric("Curvature σA", sigma_text, "local approximation"),
            _metric("Reduced χ²", f"{fit.reduced_chi_square:.4g}", f"{fit.degrees_of_freedom} residual dof", "copper"),
            _metric("Reference q max", f"{result.characteristic_q_max_nm_inv:.4g} nm⁻¹", "paper estimate"),
            _metric("Max condition", f"{np.max(fit.condition_number):.3g}", "whitened response matrix", "indigo"),
            _metric("At zero boundary", f"{100 * boundary:.1f}%", "q-dependent coefficients", "copper"),
            class_="metrics-grid",
        )

    def _result() -> ExchangeFitResult:
        state = analysis_state.get()
        req(state is not None)
        return state.result

    @output
    @render.plot
    def profile_plot():
        return plot_exchange_profile(_result())

    @output
    @render.plot
    def components_plot():
        return plot_scattering_components(_result())

    @output
    @render.plot
    def reconstruction_plot():
        return plot_reconstruction(_result())

    @output
    @render.plot
    def diagnostics_plot():
        return plot_fit_diagnostics(_result())

    @output
    @render.download(filename="mumag2022_result.zip")
    def download_result():
        state = analysis_state.get()
        req(state is not None)
        yield state.archive

    @reactive.effect
    @reactive.event(input.run_stability)
    def _run_stability() -> None:
        try:
            series = field_series()
            steps = int(input.stability_steps())
            geometry = ExperimentGeometry(str(input.geometry()))
            parameter_count = 3 if geometry is ExperimentGeometry.PERPENDICULAR else 2
            candidates = series.applied_fields_mT[: max(1, len(series.curves) - parameter_count + 1)]
            field_indices = np.unique(np.linspace(0, candidates.size - 1, min(steps, candidates.size)).astype(int))
            field_values = candidates[field_indices]
            q_min = float(input.q_min())
            q_max = float(input.q_max())
            q_values = np.linspace(max(q_min * 1.2, q_min + 1.0e-8), q_max, steps)
            with ui.Progress(min=0, max=1) as progress:
                progress.set(0.05, message="Calculating stability grid")
                result = stability_map(
                    series,
                    geometry,
                    q_max_values=q_values,
                    minimum_field_values_mT=field_values,
                    exchange_bounds_pJ_per_m=(float(input.a_min()), float(input.a_max())),
                    q_min=q_min,
                    profile_samples=21,
                )
                stability_state.set(result)
                stability_error.set(None)
                progress.set(1.0, message="Stability map complete")
        except Exception as error:
            stability_state.set(None)
            stability_error.set(str(error))

    @output
    @render.ui
    def stability_message():
        if stability_error.get():
            return ui.div(stability_error.get(), class_="calculation-message message-error")
        if stability_state.get() is None:
            return ui.div("Uses the currently selected data and fit bounds.", class_="calculation-message message-info")
        return ui.div("Stability map calculated.", class_="calculation-message message-success")

    @output
    @render.plot
    def stability_plot():
        result = stability_state.get()
        req(result is not None)
        return plot_stability_map(result)

    @reactive.effect
    @reactive.event(input.run_demag)
    def _run_demag() -> None:
        try:
            samples = int(input.demag_samples())
            with ui.Progress(min=0, max=1) as progress:
                progress.set(0.05, message="Calculating joint A/Hd surface")
                result = fit_exchange_and_demagnetizing_field(
                    field_series(),
                    ExperimentGeometry(str(input.geometry())),
                    exchange_bounds_pJ_per_m=(float(input.a_min()), float(input.a_max())),
                    demagnetizing_bounds_mT=(float(input.hd_min()), float(input.hd_max())),
                    q_range=(float(input.q_min()), float(input.q_max())),
                    minimum_applied_field_mT=float(input.field_min()),
                    exchange_samples=samples,
                    demagnetizing_samples=samples,
                )
                demag_state.set(result)
                demag_error.set(None)
                progress.set(1.0, message="Joint fit complete")
        except Exception as error:
            demag_state.set(None)
            demag_error.set(str(error))

    @output
    @render.ui
    def demag_message():
        if demag_error.get():
            return ui.div(demag_error.get(), class_="calculation-message message-error")
        if demag_state.get() is None:
            return ui.div(
                "Use this only when a common demagnetizing-field model is physically defensible.",
                class_="calculation-message message-info",
            )
        return ui.div("Joint fit calculated.", class_="calculation-message message-success")

    @output
    @render.ui
    def demag_metrics():
        result = demag_state.get()
        if result is None:
            return ui.tags.p("Calculate the surface to display the result.", class_="empty-copy")
        return ui.div(
            _metric(
                "Best A",
                f"{result.best_fit.exchange_stiffness_pJ_per_m:.4g} pJ/m",
                "joint profile",
                "indigo",
            ),
            _metric(
                "Best Hd",
                f"{result.best_demagnetizing_field_mT:.4g} mT",
                "common fitted value",
            ),
            _metric(
                "Reduced chi-square",
                f"{result.best_fit.reduced_chi_square:.4g}",
                f"{result.best_fit.degrees_of_freedom} residual dof",
                "copper",
            ),
            class_="metrics-grid demag-metrics",
        )

    @output
    @render.plot
    def demag_plot():
        result = demag_state.get()
        req(result is not None)
        return plot_demagnetizing_surface(result)


app = App(app_ui, server, static_assets=APP_DIR / "www")
