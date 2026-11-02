# Stochastic trajectory optimization with ExaModels and MadNLP

Julia port of [`../stochastic_traj_opt_jax.py`](../stochastic_traj_opt_jax.py). It
solves the same chance-constrained covariance-steering NLP with the same methodology.

The thrust chance constraints (spectral and Cholesky-slack forms) use the
executed-control covariance: feedback and navigation dispersion combined with
conditional Gates execution error. Execution uncertainty also enters UT
propagation and the covariance cost. MadNLP acceptance uses solver success and
the restoration requirement; the independently recomputed original residual is
reported as a diagnostic. Set `MADNLP_STOCHASTIC_OUTPUT_ROOT` to write a run
outside `output/`.

**Psyche** (three phases, ballistic Mars) runs through the same pipeline and the same
stochastic NLP builder, oracles and solver settings as the single-phase cases, with a
multiphase layout (`src/psyche/README.md`; formulation in
[`stochastic_optimization/README.md`](../stochastic_optimization/README.md), except for the
problem data below). It needs the exported mission arrays, which are tied to the
deterministic energy reference by SHA256:

```sh
python madnlp_stochastic_traj_opt/tools/export_psyche_inputs.py --output madnlp_stochastic_traj_opt/data/psyche_inputs.npz
julia --project=madnlp_stochastic_traj_opt --threads=4 madnlp_stochastic_traj_opt/run.jl --case=psyche --mode=cpu
julia --project=madnlp_stochastic_traj_opt --threads=4 madnlp_stochastic_traj_opt/run.jl --case=psyche --mode=gpu
```

The export refines the deterministic energy mesh by `--mesh-refinement` (default 1.5):
an integer factor splits every interval into equal pieces, a factor between 1 and 2 halves
the longest intervals of each phase, keeping every original node. The default gives
155 intervals, 56 / 54 / 45 in the Earth–Mars, Mars and Mars–Psyche phases, 101 of them powered.
Every stage uses one RK9 step per interval (UT dimension 16, as the other cases). The
energy reoptimisation keeps the closest-approach and Mars-exit epochs free; the stochastic
solve fixes them, so its durations, cost weights `N dt_k / T` and origin translations are
constants. Psyche defaults: 2500 km initial position standard deviation per axis
(`--initial-position-std-km`), terminal bound 200 km / 2 m/s per axis
(`--psyche-terminal-position-std-km`, `--psyche-terminal-velocity-std-km-s`).

The single-phase cases share with the JAX script:

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
| `gpu` | `SparseCondensedKKTSystem` (Lifted-KKT) + cuDSS LDL on a CUDA device | relaxed by `bound_relax_factor` = `--condensed-relaxation` (default `tol`/100) |

All modes build the same model; only the array backend, the kernel layout (see
[Kernels](#kernels)) and the linear algebra differ. CPU and GPU evaluations agree to
rounding (about 1e-12 relative, `test/gpu_tests.jl`). The energy-optimal seed is always
solved on the CPU.

## Quick start

From the repository root, with Julia 1.12 or newer (this validation uses 1.13).
The deterministic energy references must exist, as for the JAX script. Running
all five cases also requires the Psyche mission export shown above.

```sh
# once
julia --project=madnlp_stochastic_traj_opt -e "using Pkg; Pkg.instantiate()"

# CPU: one case, or all five in turn (--case=cr3bp selects only the three CR3BP cases)
julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl --case=lyapunov_l1_to_l2
julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl --case=all

# GPU (condensed KKT system)
julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl --mode=gpu --case=all

# Monte Carlo validation and figures alone, e.g. for an earlier solve
# (run.jl already does this after every solve; the mode also selects the JAX device)
python madnlp_stochastic_traj_opt/tools/validate.py cpu
python madnlp_stochastic_traj_opt/tools/validate.py gpu --case=nrho_l2_to_dro
python madnlp_stochastic_traj_opt/tools/validate_psyche.py madnlp_stochastic_traj_opt/output/gpu/psyche
```

After every solve, `run.jl` runs the Monte Carlo validation with the Python
validator of the case (`tools/validate.py`, or `tools/validate_psyche.py` for
Psyche) on the device of the solve: `--mode=cpu` runs JAX on every CPU core
(`JAX_NUM_CPU_DEVICES`), `--mode=gpu` runs JAX on CUDA (no preallocation, after
the solve's cached GPU memory is released); a missing GPU is an error, never a
CPU fallback. The validator Python comes from `--validation-python=PATH` or
`MADNLP_VALIDATION_PYTHON` (default `python3`) and needs JAX, e.g.
`export MADNLP_VALIDATION_PYTHON=$HOME/anaconda3/envs/project/bin/python`.
`--no-validation` skips it. A failed validation is reported with its command and
never discards the solve. All Monte Carlo draws come from `jax.random`, keyed by
seed, noise stream, arc and sample, so the CPU and GPU versions propagate the
same realizations and agree to floating-point roundoff.

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

Psyche additionally writes `psyche_nominal.npz` and `psyche_seed.npz` as the
initialisation stages finish, followed by `psyche_solution.npz` for the native
stochastic candidate. Its post-processing command is
`python madnlp_stochastic_traj_opt/tools/validate_psyche.py <result-directory>`;
this writes physical NPZ/CSV/JSON files, a continuous Monte Carlo
(by default the other cases' sample count and seed, 50 000 and 42, with 16 RK9
substeps per interval; `--samples 0` skips it) and the
figures: dispersion evolution, thrust budget, heliocentric projection and
terminal distribution in the other cases' style, plus the Mars B-plane, where
each sample is placed in the B-plane of its own osculating hyperbola at its own
closest approach.

`validate.py` imports the JAX reference, so its Monte Carlo (50 000 samples,
jax.random seed 42, 3 RK9 substeps), diagnostics and figures are exactly the reference's. It
also re-checks the Julia seed and evaluates the reference NLP residuals at the Julia
optimum with the reference's own functions and the regularisation settings saved
in the native result. The Psyche helper is separate because its event topology
and covariance scales differ from the single-phase cases.

## Status fields: what "converged" means

Following the user's 26 October revision, convergence requires MadNLP's
`SOLVE_SUCCEEDED` status and the retained restoration requirement. The independent
original-constraint residuals are still reported, but no additional 1e-8 residual
acceptance test is performed. The existing default solver tolerance remains 1e-6.

| Field | Meaning |
|---|---|
| `madnlp_status`, `solver_success` | MadNLP's own termination at its configured scaled KKT tolerance |
| `restoration_requirement_met` | No run of more than five consecutive restoration iterations |
| `max_constraint_violation` | Independently recomputed original residual, diagnostic only |
| `original_residual_acceptance_check` | `false` under the current policy |
| `converged` | Solver success and restoration requirement |
| `final_kkt` | Scaled and unscaled final KKT measures |

The same acceptance policy applies to CPU and GPU and to the external Python
validation/export helper. GPU equality relaxation remains part of the configured
solver model and its actual original residual remains visible in the report.
Historical reports retain the acceptance policy used when they were generated.
Psyche additionally retains its strict closest-approach domain check: positive
distance curvature and positive hyperbolic excess energy. This is not an
original-residual acceptance threshold.

## Defaults

The shared regularisation settings now match the current JAX defaults and are
the same in both execution modes. Older comparisons used smaller spectral
regularisation and an unsmoothed sigma-point mass-flow norm; those are historical
settings, not the defaults of the present comparison.

| Setting | Default | JAX reference |
|---|---|---|
| control norm ε in `sqrt(‖S‖² + ε²)` (seed and stochastic solves) | 1e-5 | 1e-5 |
| box bound on the scaled feedback gains, `|K̂| ≤ B` (`--gain-bound`, `Inf` disables) | 50 | 50 |
| weight `w` of `tr(T)` (control covariance) in the cost and TVLQR seed `R~ = w I` (`--control-covariance-weight`) | Lyapunov 33.56, Halo 118.4, NRHO 129.2, Earth–Mars 151.8, Psyche 1000 | same |
| weight `w_P` of `tr(P)/c²` (state covariance) in the cost and TVLQR seed `Q~ = w_P I/c²` (`--state-covariance-weight`) | Lyapunov 0.01994, Halo 0.2128, NRHO 0.1531, Earth–Mars 0.1530, Psyche 1 | same |
| sigma-point mass flow `T_max·sqrt(‖e‖² + δ²)` | δ = ε | δ = ε |
| eigenvalue smoothing `s` in `p = sqrt(p2/6 + s²)` | 1e-6 | 1e-6 |
| spectral radius floor `f` in `sqrt(λ_max + f²)` | 1e-6 | 1e-6 |
| terminal margin factor diagonal floor | 1e-5 | 1e-5 |

Solver settings of the stochastic solve:

| Setting | Default |
|---|---|
| `tol` = `acceptable_tol` | 1e-6 (SNOPT reference: optimality 1e-6, feasibility 1e-8) |
| `max_iter` | 5000 |
| `alpha_min_frac` | 0.05 |
| barrier | monotone, μ from 0.1 down to 1e-10 |
| consecutive restoration limit | none (the ≤ 5 requirement is still checked) |
| `gpu`: `bound_relax_factor` | `tol`/100 = 1e-8 |
| `cpu`: `bound_relax_factor` | 0 (exact bounds) |

The energy-optimal seed is solved with `tol` = 1e-8, exact equalities and MUMPS.

## Historical results

The tables below predate the current executed-covariance formulation and
acceptance policy; they are kept for reference.

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

| JAX (`stochastic_traj_opt_jax.py`) | Julia (`src/`) |
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
- **UT moments.** An `ExaModels.add_eval` oracle on the native defect rows (one per set
  of UT parameters; Psyche has two). The Hessian is
  `Σ_s W_sᵀ [H_s W_s + 2w J_sᵀ N (V_s − V̄)] + ∇²ζ`, built from per-sigma-point
  forward-over-reverse RK9 Hessians (a hand-written discrete adjoint evaluated with dual
  numbers). Every term is exact. The Gauss–Newton term uses the centered `V_s − V̄`, and
  the deviations `y_s − ȳ` and `V_s − V̄` use corrected two-pass means, so the
  cancellations that make these sums vanish are exact to rounding of the deviations
  rather than of the states: against a 128-bit reference the Hessian is accurate to
  1e-14 (Earth–Mars) and 1e-12 (Psyche); the previous assembly lost up to 5e-6 in its
  mean–mean block. The state spread is the node factor itself, `S = √d diag(std) L`
  (`S Sᵀ = d P`; the JAX script refactorizes `chol(d P + 1e-12 I)`, equal to 1e-12), so
  the pre-map curvature `∇²ζ` comes only from the executed controls: per outer sigma
  point the gradient and Hessian of the Gates sum in the 3-vector command, contracted
  with the closed-form command Jacobian and its bilinear gain–factor cross terms.
- **Thrust chance constraint.** A `VectorNonlinearOracle` with the smoothed
  trigonometric 3×3 eigenvalue formula applied to the closed-form `T`.

## Kernels

All cases, Psyche included, use the same kernels. All derivative kernels are
KernelAbstractions kernels (threads on the CPU, CUDA on the GPU). The per-arc kernels
(moments, reductions, Jacobian rows, multipliers, command curvature and ζ) are shared. The per-sigma-point derivatives
(`W_s`, `J_s`, `H_s` and the products `V_s = J_s W_s`, `B_s`) are organized per backend
in `oracles/ut_moments.jl`:

| | CPU | GPU |
|---|---|---|
| Sigma-point derivatives | one thread per sigma point, dual numbers over all 56 arc inputs | stages split over threads: `W_s` in chunks of 8 of the 56 directions, `J_s` and `H_s` per point, `V_s` and `B_s` one column per thread |
| `Σ_s W_sᵀ B_s` | one entry per thread | shared-memory tiles |

On the same device the two layouts give bitwise identical results: every partial
derivative comes from the same operations at any chunk width, and the assembly sums in
the same order. The 10 sigma-point directions of the RK9 step are always seeded
together: StaticArrays' `dot` sums with `@simd`, so a narrower dual would change the
summation order. GPU launches are asynchronous (kernels on one stream run in order;
copies to the host synchronize).

## Compilation and performance measurements

`run.jl` loads the local `CR3BPStochastic` Julia package. The first load builds
Julia's package cache; later processes reuse it. `src/precompile_workload.jl`
uses small synthetic CR3BP and heliocentric models. It does not run an optimiser, read mission data, or create a CUDA
context. A source/dependency/Julia-version change can invalidate that cache.
GPU kernels and mission-specific model construction can still compile at run
time; package precompilation does not eliminate GPU startup costs.

The commands below measure fixed candidates without running optimisation or
overwriting production results. Run them sequentially on an otherwise idle
GPU. `OPENBLAS_NUM_THREADS=1` avoids nested BLAS threading in these measurements.

```sh
OPENBLAS_NUM_THREADS=1 julia --project=madnlp_stochastic_traj_opt --threads=4 madnlp_stochastic_traj_opt/tools/profile_gpu.jl --case=psyche --repeats=5
OPENBLAS_NUM_THREADS=1 julia --project=madnlp_stochastic_traj_opt --threads=4 madnlp_stochastic_traj_opt/tools/benchmark_startup.jl source
OPENBLAS_NUM_THREADS=1 julia --project=madnlp_stochastic_traj_opt --threads=4 madnlp_stochastic_traj_opt/tools/benchmark_startup.jl package
```

`profile_gpu.jl` also accepts each CR3BP case and `earth_mars`. It measures
first-call and warmed CPU/GPU evaluations, checks numerical agreement, and times initial KKT assembly
and factorisation. CUDA events and explicit synchronization distinguish device
elapsed time from CPU submission time. CUPTI activity traces record transfers,
synchronization, device gaps, and kernel resource usage. Initial-candidate KKT
timings do not predict factorisation cost at every later iterate.

Results go to `output/profile/<case>/` unless `--output` is given. On the RTX 5060
Laptop GPU, the cached package reduced startup from about 103 s (source include
plus first workloads) to about 2 s after a one-time 100 s cache build. These
checks establish numerical equivalence of evaluations, not trajectory convergence.

## Verification commands

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
- `psyche_tests.jl` (needs the mission export): uncertainty model, ballistic Mars,
  B-plane Jacobian against ForwardDiff, the Psyche model against the literal arc map,
  directional central differences of its Jacobian and Lagrangian Hessian.
- `gpu_tests.jl`, `psyche_gpu_tests.jl`: CUDA objective, gradient, constraint, Jacobian
  and Hessian entries against the CPU model.
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
