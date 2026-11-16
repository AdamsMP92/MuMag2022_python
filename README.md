# MuMag2022 Python

MuMag2022 Python is a numerically revised, browser-accessible implementation
of the field-dependent magnetic SANS analysis introduced in

> M. P. Adams, M. Bersweiler, E. M. Jefremovas, and A. Michels,
> *J. Appl. Cryst.* **55**, 1055–1062 (2022),
> https://doi.org/10.1107/S1600576722005349.

For a series of azimuthally averaged unpolarized SANS curves measured in the
approach-to-saturation regime, the perpendicular-geometry model is

```text
I(q, H) = I_res(q) + S_H(q) R_H(q, H; A) + S_M(q) R_M(q, H; A),
```

while the parallel geometry omits the `S_M R_M` contribution. At each `q`,
the non-negative linear coefficients are determined directly by weighted
non-negative least squares. The exchange stiffness `A` is then profiled as a
global nonlinear parameter.

## What is different from the original prototype?

- The whitened design matrix is fitted directly; normal equations are not
  used, avoiding an unnecessary squaring of the condition number.
- Input units, common q grids, positive uncertainties, and positive internal
  fields are validated explicitly.
- Rank, condition number, active constraints, standardized residuals, and the
  complete exchange-stiffness profile are returned with every fit.
- Local curvature errors are labelled as approximations. Profile intervals
  and stability diagnostics make their assumptions visible.
- Stability maps quantify the dependence of the fitted exchange stiffness on
  `q_max` and the minimum applied field.

## Installation and tests

The development environment is managed with [Pixi](https://pixi.sh):

```text
pixi run test
```

Run the local browser application with

```text
pixi run web
```

The deployed Shinylive application performs the analysis locally in the
browser; uploaded data are not sent to a server.

A minimal script is available in [`examples/synthetic_fit.py`](examples/synthetic_fit.py).
The numerical decisions, diagnostics, and uncertainty limitations are
described in [`docs/numerical_method.md`](docs/numerical_method.md).

## Legacy input format

Each field-dependent curve is a headerless three-column text file containing

```text
q [nm^-1]    I(q) [cm^-1 or arb. units]    sigma_I(q)
```

with filename

```text
index_applied-field_mT_saturation-magnetization_mT_demagnetizing-field_mT.csv
```

for example `3_61_1640_23.748285.csv`. The field and magnetization values are
the corresponding `mu_0`-scaled quantities in mT.

## Scope and assumptions

The method is intended for unpolarized, azimuthally averaged SANS data from
statistically isotropic bulk ferromagnets in the approach-to-saturation
regime. It assumes a common exchange stiffness, positive internal fields,
field-independent scattering functions over the selected field range, and
neglects chiral terms. A numerically successful fit does not establish these
physical assumptions; the stability and residual diagnostics are intended to
make their consequences visible.

## Citation

When using MuMag2022 Python in research, please cite the exact software
version and the associated 2022 *Journal of Applied Crystallography* article.
GitHub exposes the software citation from [`CITATION.cff`](CITATION.cff); a
version-specific DOI will be provided by the Zenodo archive linked to the
corresponding GitHub release.

## License

MIT. See [LICENSE](LICENSE).
