# Formulation and numerical method

The defaults reproduce the Python data: 20 arcs, 0.25 s per arc, 5 s final time,
four state components `[px, py, vx, vy]`, and two control components. Options
validate `tf == n_arcs * dt`.

Eigenvalue smoothing defaults to `1e-5` instead of Python's
`1e-6` to reduce curvature near isotropic control covariances. Set
`spectral_eigenvalue_smoothing=1e-6` in `Options` to recover the original value.
The requested upper limit of `1e-4` is enforced.
The selected value and radius bias are saved in the diagnostics. The Python
verification uses that same value for an exact formulation comparison.

| Decision-variable block | Count |
|---|---:|
| Node means | `4(N+1)` |
| Feedforward controls | `2N` |
| Feedback gains | `8N` |
| Packed node Cholesky factors | `10(N+1)` |
| Packed terminal margin factor | `10` |
| **Total** | **`24N+24 = 504`** |

There are `14N+28 = 308` equality constraints and `7N+8 = 148` inequality
constraints. The diagonal restrictions remain constraint rows as in Python.
Endpoint variables are retained and constrained by equalities. Model construction
asserts these counts. ExaModels subexpressions are inlined and introduce no
auxiliary optimization variables. MadNLP's internal slack variables and dual
variables are not part of this count.

`ut_moment_oracle.jl` and `control_chance_oracle.jl` implement ExaModels oracle
callbacks for the nested UT moments and control chance margins. Expanding these
expressions into symbolic trees causes excessive compilation. The callbacks use
KernelAbstractions loops and nested ForwardDiff dual numbers for exact sparse
Jacobians and Lagrangian Hessians on the selected device. They augment existing
moment constraint rows or register chance constraint rows, without adding
decision variables. The remaining model uses ExaModels' symbolic expressions.

Packed lower-triangular order is the Python row order:
`(1,1), (2,1), (2,2), (3,1), (3,2), (3,3), (4,1), (4,2), (4,3), (4,4)`.
Julia's internal array storage order differs from NumPy's. Saved NPZ arrays use
the Python axis conventions, including `gains[N,2,4]` and
`state_covariances[N+1,4,4]`.

The augmented UT uses `z = [x; xi; eta]`, covariance `diag(P,I4,I4)`, 25 sigma
points, and the original kappa weights. `W = sqrt(dt)*xi` and
`H = sqrt(dt/12)*eta`. Each sigma-point control is
`u = feedforward + K*(x_sigma - mean)` and is held constant over the arc.
The state and control moments are reconstructed from the propagated sigma points.
No linear covariance recurrence is used in the optimization.

For efficient sparse differentiation, the model propagates centered sigma offsets
through the same ShARK stages. Because this particular propagation map is affine,
each positive/negative sigma pair stays symmetric and the weighted offset mean
is zero. All 25 points contribute to the state moment constraints. For the
control covariance, the central and noise-only sigma points have zero control
offset, so only the eight nonzero state-offset points contribute. These are
algebraic properties of the original UT; the separate numeric implementation
reconstructs the full absolute sigma cloud for independent verification.

The ShARK tableau has `a21=5/6`, drift weights `(2/5,3/5)`, Brownian stage
coefficients `(0,5/6)`, and area stage coefficients `(1,1)`. The script implements
these stages for the double-integrator drift with constant additive diffusion.
The continuous diffusion is calibrated by inverting the same finite-horizon
Lyapunov operator as Python. Because `Ac^2=0`, the covariance integral can be
evaluated exactly instead of numerically integrated. Analytic linear moment
propagation is used only for independent tests.

The deterministic seed minimizes squared control energy, with the original
endpoint and nominal path constraints. The stochastic objective instead uses
the regularized **control norm**, plus `tr(Q*P)` and `tr(R*S)`, all multiplied
by `dt`. It has no extra mean-state tracking cost or terminal cost.


## Initialization and terminal constraints

The nominal transfer minimizes squared control effort with the original mean
endpoint and nominal path constraints. All gain entries default to positive
`1e-5`. Means and covariances are then repropagated with the full UT-ShARK map.
This enforces shooting dynamics in the seed; it does not promise initial
chance-constraint or terminal-covariance feasibility. The terminal margin seed
uses the original positive-semidefinite projection and squared diagonal floor.

TVLQR initialization remains optional. Its Bryson state weights are reciprocal
initial variances, its terminal weights reciprocal target variances, and its
control weight is `tvlqr_control_weight_scale * I / u_max^2`. The retained
optional scale defaults are 1 for CPU/condensed and 1000 for hybrid; they have
no effect on uniform initialization. Set the scale to 1 and disable uniform
initialization to recover the original Python TVLQR seed.

Path chance constraints apply at nodes `0,...,N-1`. Control chance constraints
retain the chi-square quantile and smoothed 2-by-2 maximum eigenvalue. Terminal
headroom is represented by `I - D*P_N*D = M*M'`, with
`D = diag(1/sqrt(diag(P_target)))` and `diag(M) >= 1e-7`. State Cholesky
nonnegativity remains in the original constraint rows. No redundant variable
bounds or extra decision variables are introduced.

## KKT systems and restoration

CPU uses `MadNLP.SparseKKTSystem` and MUMPS with exact equalities. GPU condensed
uses `MadNLP.SparseCondensedKKTSystem` and cuDSS LDL with bound-relaxation factor
`1e-9`. This factor is not an absolute equality-error bound: the pinned MadNLP
implementation scales finite bounds and relaxes constraint and slack bounds.
Final acceptance always uses original, unrelaxed constraints.

GPU hybrid enforces equalities directly through the existing regularization
adapter around `HybridKKT.HybridCondensedKKTSystem`, with cuDSS LDL, conjugate
gradients, and gamma `1e7`. See [hybrid_kkt.md](hybrid_kkt.md) for the linear
algebra and why the adapter is needed by the pinned package version.

All modes default to `alpha_min_frac=0.05`, initial stochastic barrier `0.01`,
and at most two consecutive restoration iterations. The nominal solve uses
barrier `0.1` and a tolerance capped at `1e-8`. Soft and robust restoration
steps share one consecutive-step counter; the guard blocks a third step
and rejects that solve. An unsuccessful regular step does not reset the
restoration count. These overrides apply only to solvers carrying this
project's `RestorationGuard`.

The default is exactly one stochastic solve. The pre-existing explicit
`condensed_continuation=true` option remains available for compatibility; it
is disabled by default and was not used for current validation. Its staged
iteration budget and restoration accounting remain shared.

## References

- [MadNLP documentation](https://madsuite.org/MadNLP.jl/stable/)
- [MadNLP GPU tutorial](https://madsuite.org/MadNLP.jl/stable/tutorials/gpu/)
- [ExaModels guide](https://madsuite.org/ExaModels.jl/stable/guide/)
- [Diffrax ShARK implementation](https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_solver/shark.py)
- [Foster, dos Reis and Strange](https://arxiv.org/abs/2210.17543)
