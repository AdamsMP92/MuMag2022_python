# Numerical method and interpretation

## Model structure

For each selected momentum transfer `q`, MuMag2022 writes the field series as

```text
perpendicular: I = I_res + S_H R_H(A) + S_M R_M(A)
parallel:      I = I_res + S_H R_H(A)
```

The response functions depend on the magnetic field, saturation
magnetization, exchange stiffness `A`, and geometry. The scattering functions
`I_res(q)`, `S_H(q)`, and, in perpendicular geometry, `S_M(q)` are constrained
to be non-negative.

## Direct weighted solution

At fixed `A`, the code divides each row of the design matrix and each measured
intensity by its supplied standard uncertainty. It then solves the whitened
system directly by non-negative least squares:

```text
minimize ||X_w beta - y_w||_2, subject to beta >= 0.
```

The normal equations are deliberately not formed. Solving `X.T X beta = X.T
y` would square the condition number and can discard useful numerical
precision. Conditional covariance estimates for coefficients away from the
zero boundary are likewise calculated from the SVD pseudoinverse of `X_w`,
not from an explicit inverse of `X_w.T X_w`.

## Global exchange-stiffness profile

The exchange stiffness is nonlinear but common to every selected `q`. The
code therefore evaluates the total chi-square on a user-visible global grid
in `A`, refines the best interior grid cell with a bounded scalar optimizer,
and returns the complete sampled profile. This guards against interpreting a
single local optimization result without seeing the objective function.

The reported reduced chi-square subtracts the active linear coefficients and
one additional degree of freedom for the profiled global value of `A`.

## Diagnostics

Every fit exposes:

- rank and condition number of the whitened response matrix at each `q`;
- coefficients at the non-negative boundary;
- raw and standardized residuals;
- the sampled `A` profile and approximate profile intervals;
- a stability map under changes of `q_max` and minimum applied field.

These diagnostics distinguish numerical solvability from physical
identifiability. A large condition number means that different scattering
components produce nearly indistinguishable field dependences. A coefficient
at zero has a boundary distribution, so a symmetric covariance error is not
an ordinary confidence interval for that coefficient.

The optional joint fit replaces all file-specific demagnetizing fields by one
common fitted value and samples the two-dimensional `A`/`H_d` surface before a
bounded local refinement. It is an identifiability diagnostic, not a reason to
assume that one common demagnetizing field is physically appropriate.

## Uncertainty limitations

The local curvature error and the delta-chi-square profile intervals are
asymptotic approximations. They can be unreliable when constraints are active,
the optimum touches the `A` search boundary, residuals are correlated, or the
input error bars are not calibrated independent Gaussian standard
deviations. The optional parametric bootstrap tests estimator variability
under the fitted model and supplied Gaussian errors; it does not validate
those assumptions.

## Data-selection limitations

The physical response model is intended for the approach-to-saturation
regime. The selected field range and `q` range are therefore part of the
scientific model, not merely plotting choices. A robust result should persist
over a defensible neighborhood in the `q_max`/minimum-field stability map and
should not depend on severely ill-conditioned response matrices.

