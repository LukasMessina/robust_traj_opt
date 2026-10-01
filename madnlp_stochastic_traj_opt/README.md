# CR3BP stochastic trajectory optimization with ExaModels and MadNLP

Julia port of [`../cr3bp_stochastic_traj_opt_jax.py`](../cr3bp_stochastic_traj_opt_jax.py). It
solves the same chance-constrained covariance-steering NLP with the same methodology:

- the same normalization and nondimensionalization;
- the augmented unscented transform with navigation error, XMDS2 RK9 acceleration
  diffusion and nested Gates execution error;
- the energy-optimal + Bryson TVLQR + UT-propagated-covariance initial guess;
- the same decision variables (56N + 28) and constraints (36N + 34).

The model is built with ExaModels and solved by MadNLP with **exact Hessians only**, in
one of two execution modes:

| Mode | KKT system and linear solver | Equality constraints |
|---|---|---|
| `cpu` (default) | `SparseKKTSystem` + MUMPS | exact |
| `gpu` | `SparseCondensedKKTSystem` (Lifted-KKT) + cuDSS LDL on a CUDA device | relaxed by `bound_relax_factor` = `--condensed-relaxation` (default `tol`/10) |

Both modes build the same model; only the array backend, the kernel layout (see
[Kernels](#kernels)) and the linear algebra differ. CPU and GPU evaluations agree to
rounding (about 1e-12 relative, `test/gpu_tests.jl`). The energy-optimal seed is always
solved on the CPU.

## Quick start

From the repository root, with Julia 1.12. The deterministic references
`output/cr3bp_deterministic_traj_opt/energy_optimal/<case>.npz` must exist, as for the
JAX script.

```sh
# once
julia --project=madnlp_stochastic_traj_opt -e "using Pkg; Pkg.instantiate()"

# CPU: one case, or all three in turn
julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl --case=lyapunov_l1_to_l2
julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl --case=all

# GPU (condensed KKT system)
julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl --mode=gpu --case=all

# Monte Carlo validation and figures with the reference code (Python, JAX on the CPU)
python madnlp_stochastic_traj_opt/tools/validate.py cpu
python madnlp_stochastic_traj_opt/tools/validate.py gpu --case=nrho_l2_to_dro
```

`run.jl --help` lists every option. The first run of a session compiles the derivative
kernels (a few minutes, included in the reported times). On an 8 GB laptop GPU, run one
GPU process at a time: two processes oversubscribe the device and WDDM paging stalls both.

## Outputs

Every file of a run goes to `output/<mode>/<case>/` (`<mode>` is `cpu` or `gpu`); a new
run of the same case and mode replaces it.

| File | Content |
|---|---|
| `<case>.npz` | Solution arrays with the JAX key names where they apply, plus the reference trajectory, node scales and seed data used by `validate.py` |
| `<case>_diagnostics.toml` | Solver status, final KKT measures, restoration statistics, independently recomputed residuals, settings and timings |
| `<case>_madnlp_nominal.log` | MadNLP log of the energy-optimal seed solve |
| `<case>_madnlp_stochastic.log` | MadNLP log of the stochastic solve |
| `validation/` | Written by `tools/validate.py`: the reference's NPZ/CSV export, the four figures (`*_dispersion_evolution.png`, `*_thrust_budget.png`, `*_projections.png`, `*_terminal_distribution.png`) and `*_validation.json` |

`tools/summarize_runs.py [cpu|gpu]` tabulates the diagnostics of a mode into
`output/<mode>/summary.csv`.

`validate.py` imports the unchanged JAX script, so its Monte Carlo (50 000 samples,
NumPy seed 42, 3 RK9 substeps), diagnostics and figures are exactly the reference's. It
also re-checks the Julia seed and evaluates the reference NLP residuals at the Julia
optimum with the reference's own functions (unsmoothed eigenvalue formula, plain ‖u‖
sigma-point mass flow).

## Status fields: what "converged" means

MadNLP stops on a single tolerance of its *scaled* KKT error, while the reference
accepts a solution only when SNOPT succeeds **and** every original residual is at most
1e-8. The report keeps these apart:

| Field | Meaning |
|---|---|
| `madnlp_status`, `solver_success` | MadNLP's own termination (`SOLVE_SUCCEEDED`: scaled KKT error ≤ `tol`) |
| `restoration_requirement_met` | No run of more than 5 consecutive restoration iterations. By default there is no limit and the requirement is only checked; with `--max-restoration=5` a 6th consecutive restoration step is blocked and the solve stops with `USER_REQUESTED_STOP`. |
| `max_constraint_violation`, `jax_feasibility_met` | Largest original (unscaled, unrelaxed) residual, and whether it is ≤ 1e-8 |
| `converged` | All three: the reference's acceptance criterion |
| `final_kkt` | MadNLP's final dual infeasibility, constraint violation and complementarity, scaled and unscaled |

The process exit status is 1 when a run does not end with `SOLVE_SUCCEEDED` or breaks
the restoration requirement. In `gpu` mode the equalities are relaxed by
`bound_relax_factor`, so their residuals end near that value and `converged` is
normally `no`.

## Defaults

Problem data are the JAX defaults. These robustification settings differ from the
reference and are the same in both modes:

| Setting | Default | JAX reference |
|---|---|---|
| control norm ε in `sqrt(‖S‖² + ε²)` (seed and stochastic solves) | 1e-6 | 1e-6 |
| sigma-point mass flow `T_max·sqrt(‖e‖² + δ²)` | δ = ε | δ = 0 (plain ‖u‖) |
| eigenvalue smoothing `s` in `p = sqrt(p2/6 + s²)` | 1e-6 | 1e-15 (literal 1e-30 for s²) |
| spectral radius floor `f` in `sqrt(λ_max + f²)` | 1e-6 | 1e-7 |
| terminal margin factor diagonal floor | 1e-5 | 1e-4 |

Solver settings of the stochastic solve:

| Setting | Default |
|---|---|
| `tol` = `acceptable_tol` | 1e-6 (SNOPT reference: optimality 1e-6, feasibility 1e-8) |
| `max_iter` | 5000 |
| `alpha_min_frac` | 0.05 |
| barrier | monotone, μ from 0.1 down to 1e-8 |
| consecutive restoration limit | none (the ≤ 5 requirement is still checked) |
| `gpu`: `bound_relax_factor` | `tol`/10 = 1e-7 |
| `cpu`: `bound_relax_factor` | 0 (exact bounds) |

The energy-optimal seed is solved with `tol` = 1e-8, exact equalities and MUMPS.

## Results with the defaults

CPU (`--threads=auto`, 20 threads; CPU runs are deterministic and independent of the
thread count):

| Case | Arcs | Variables | MadNLP | Iterations | Objective (SNOPT) | Max violation | Dual inf. | Longest restoration run | Converged | Solve |
|---|---:|---:|---|---:|---|---:|---:|---:|---|---:|
| Lyapunov L1 → L2 | 48 | 2716 | SOLVE_SUCCEEDED | 1016 | 6.5948994630 (6.6955166) | 9.5e-7 | 4.2e-13 | 0 | no (> 1e-8) | 139 s |
| NRHO L2 → DRO | 85 | 4788 | SOLVE_SUCCEEDED | 4589 | 44.1199870937 (44.8889798) | 1.3e-8 | 4.9e-7 | 2 | no (> 1e-8) | 912 s |
| Halo L2 → L1 | 80 | 4508 | SOLVE_SUCCEEDED | 2335 | 50.8050211660 (51.0908983) | 8.2e-7 | 9.2e-13 | 1 | no (> 1e-8) | 429 s |

Complementarity ends at about 1e-8 (the barrier floor). The iteration path is
sensitive to tiny perturbations of the problem data, so a changed setting can change
the outcome more than its size suggests.

GPU (RTX 5060 Laptop, `bound_relax_factor` = 1e-7):

| Case | MadNLP | Iterations | Objective | Max violation | Longest restoration run | Solve |
|---|---|---:|---|---:|---:|---:|
| Lyapunov L1 → L2 | SOLVE_SUCCEEDED | 681 | 6.5915258643 | 2.4e-7 | 6 | 203 s |
| NRHO L2 → DRO | SOLVE_SUCCEEDED | 1479 | 44.1126500365 | 2.0e-7 | 0 | 211 s |
| Halo L2 → L1 | SOLVE_SUCCEEDED | 1531 | 50.7978019210 | 2.1e-7 | 20 | 285 s |

The relaxed equalities and bounds leave residuals of about 2 × `bound_relax_factor`, so
no GPU run meets 1e-8, and the objectives are slightly below the CPU ones. GPU runs are
not bit-reproducible (the model evaluations are, but the GPU linear algebra is not):
repeated runs follow slightly different paths to the same optimum (objectives agree to
about 1e-13), with different iteration and restoration counts.

`validate.py` on all six runs, with the reference code:
- seed gains within 1.3e-15 and propagated seed covariances within 1.4e-10 of the reference;
- the reference objective equals the Julia one to all printed digits;
- no chance or bound violation;
- Monte Carlo worst-arc thrust violation frequency of 0.072–0.084 % (limit 1 %).

The reference NLP residuals (2.7e-7 to 9.5e-7) include the effect of the robustification
settings, since the reference functions use the plain ‖u‖ mass flow and the unsmoothed
eigenvalue formula.

## Method

| JAX (`cr3bp_stochastic_traj_opt_jax.py`) | Julia (`src/`) |
|---|---|
| `load_ref_traj`, `_build_uniform_ref_traj` (incl. `np.linspace`, `np.interp`) | `reference.jl` |
| `reoptimize_reference_trajectory` (Dopri8 shooting, SNOPT) | `nominal.jl` (MadNLP/MUMPS) |
| `Normalization`, navigation covariance, diffusion units | `problem.jl` |
| `integration_map` (Dopri8), `stochastic_integration_map` (XMDS2 RK9) | `dynamics.jl`, tableaux in `generated/` |
| `build_propagation_arc_map` (33 outer, 198 nested sigma points) | `unscented.jl` (literal), `oracles/ut_moments.jl` (kernels) |
| `normalized_arc_jacobians`, `tvlqr_gains`, `build_initial_guess` | `initialization.jl` |
| seeded `TrajectoryNLP` | `model.jl`, `oracles/ut_moments.jl`, `oracles/chance.jl` |
| `recover_solution` | `diagnostics.jl` (literal arc map, independent of the kernels) |
| `monte_carlo_rollouts`, `compute_diagnostics`, `save_outputs` | reused as is by `tools/validate.py` |

The model has exactly the JAX variables, **56N + 28** (node means, normalized
feedforward, scaled gains, seed-scaled node Cholesky factors, terminal margin factor),
the **36N + 34** constraints and the same bounds; construction asserts both counts.

Exact second derivatives:

- **Objective.** Native ExaModels algebra (ExaModels 0.12 objective oracles have no
  Hessian path). The nested-UT executed-control covariance has an exact closed form,
  `T = (1/d)[(1+σ₂²−σ₄²) K M Kᵀ + σ₄² tr(K M Kᵀ) I] + Q_G(f) + εI` with
  `M = d P₆ + εI + d R_nav`; it agrees with the literal 198-point sum to about 1e-15.
- **UT moments.** An `ExaModels.add_eval` oracle on the native defect rows. The
  Hessian is `Σ_s W_sᵀ (H_s + 2w J_sᵀ N J_s) W_s − 2 V̄ᵀ N V̄ + ∇²ζ`, built from
  per-sigma-point forward-over-reverse RK9 Hessians (a hand-written discrete adjoint
  evaluated with dual numbers) and pair-seeded pre-map curvature.
- **Thrust chance constraint.** A `VectorNonlinearOracle` with the smoothed
  trigonometric 3×3 eigenvalue formula applied to the closed-form `T`.

## Kernels

All derivative kernels are KernelAbstractions kernels (threads on the CPU, CUDA on the
GPU). The per-arc kernels (moments, spread-factor derivative, reductions, Jacobian rows,
multipliers, pre-map curvature ζ) are shared. The per-sigma-point derivatives
(`W_s`, `J_s`, `H_s` and the products `V_s = J_s W_s`, `B_s`) are organized per backend
in `oracles/ut_moments.jl`:

| | CPU | GPU |
|---|---|---|
| Sigma-point derivatives | one thread per sigma point, dual numbers over all 56 arc inputs | stages split over threads: `W_s` in chunks of 8 of the 56 directions, `J_s` and `H_s` per point, `V_s` and `B_s` one column per thread |
| Spread-factor derivative | all 28 directions per thread | 4 directions per thread |
| `Σ_s W_sᵀ B_s` | one entry per thread | shared-memory tiles |

On the same device the two layouts give bitwise identical results: every partial
derivative comes from the same operations at any chunk width, and the assembly sums in
the same order. The 10 sigma-point directions of the RK9 step are always seeded
together: StaticArrays' `dot` sums with `@simd`, so a narrower dual would change the
summation order. GPU launches are asynchronous (kernels on one stream run in order;
copies to the host synchronize).

## Verification

```sh
julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/test/runtests.jl        # CPU
julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/test/runtests.jl --gpu                 # + CUDA
```

- `configuration_tests.jl`: command-line parsing, the execution modes and the output layout.
- `dynamics_tests.jl`: vector-field VJP, discrete RK adjoint, and forward-over-reverse
  Hessians against central differences (both tableaux).
- `model_tests.jl` (3-arc problem): oracle values against the literal arc map
  (1e-12), objective against the literal nested UT, Jacobian and full Lagrangian
  Hessian against central differences.
- `gpu_tests.jl`: CUDA objective, gradient, constraint, Jacobian and Hessian entries
  against the CPU model.
- `tools/oracle_snapshot.jl`: bitwise regression check of every model and oracle
  evaluation at three fixed points against a saved snapshot.
- `tools/validate.py`, against the JAX reference: TVLQR gains and gain scales to
  about 1e-15, propagated seed covariances to about 1e-10, and the reference NLP
  residuals and objective at the Julia optimum.

## Layout

```text
run.jl                       command-line entry point
src/CR3BPStochastic.jl       module
  options.jl                 configuration, defaults, validation, output directory
  cli.jl                     command-line parsing and the run loop
  generated/                 tableaux and case constants (tools/export_reference_data.py)
  packing.jl problem.jl dynamics.jl unscented.jl
  reference.jl nominal.jl initialization.jl model.jl
  oracles/                   dual-number seeds and kernel launch, UT-moment and chance oracles
  solver/                    MadNLP settings per mode, restoration guard
  diagnostics.jl solve.jl output.jl
test/                        runtests.jl (+ --gpu)
tools/
  validate.py                reference Monte Carlo, diagnostics and figures
  summarize_runs.py          output/<mode>/summary.csv from the diagnostics
  restoration_diagnosis.jl   anatomy of a restoration phase (multipliers, Hessian and Jacobian blocks)
  oracle_snapshot.jl         bitwise regression check of the NLP evaluations
  profile_oracles.jl         timing of the oracle callbacks and model evaluations
  export_reference_data.py   regenerate src/generated/ from the Python sources
output/cpu/<case>/  output/gpu/<case>/
```
