# Stochastic trajectory optimization with ExaModels and MadNLP

Chance-constrained covariance steering of low-thrust transfers, in the model of
[`../stochastic_traj_opt_jax.py`](../stochastic_traj_opt_jax.py): node means, throttle and unit direction,
feedback gains and node covariance factors, an augmented unscented transform with navigation error,
XMDS2 RK9 acceleration diffusion and nested Gates execution error, and thrust chance constraints on the
executed control (feedback and navigation dispersion combined with the conditional execution error).

Five cases: Lyapunov L1 → L2, Halo L2 → L1, NRHO L2 → DRO (CR3BP), Earth–Mars (heliocentric) and Psyche
(three phases with a ballistic Mars flyby; `src/psyche/README.md`). There is one formulation and two problems:

| | Step 3 (`--objective=standard`, default) | Step 4 (`--objective=chance_sum`) |
|---|---|---|
| Cost | `Σ w_k (Γ_k + w tr T_k) + w_P Σ w_k tr P_k / c² + tr(G P_N Gᵀ)/c²` | `Σ w_k (Γ_k + ψ s_k τ̂_k)`: the time-weighted 99 % bound of the executed thrust |
| Thrust chance rows | `Γ_k + ψ ρ(T_k) ≤ 1`, ρ the smoothed largest-eigenvalue formula | `Γ_k + ψ s_k τ̂_k ≤ 1` with the epigraph `τ̂_k² I − Ĝ_k Ĝ_kᵀ − T_k/s_k² = 0` |
| Terminal covariance | in the cost (Bryson weight) | hard bound `G P_N Gᵀ ≤ I` |
| TVLQR seed weights | the cost's `w_P`, `w` | Bryson's rule, 1 and 1 (Psyche: the cost's) |
| MadNLP iteration budget | 5000 | 10000 |

Common to both: the deterministic optimisation's controls (throttle `Γ_k ∈ [0, 1]`, direction `d_k` with
`‖d_k‖² = 1`, thrust `Γ_k d_k`), the fuel-optimal reference (single-phase: its own mesh with every interval
split in two, arc weights `N dt_k / T`; Psyche: the exported fuel-optimal mission, mesh refined ×2),
square-root covariance defects `L_{k+1} − chol(V_k)`, and the sigma-point mass flow
`T_max sqrt(‖u‖² + δ²)` with δ = 5e-4 on the 24 outer points whose command carries feedback (the exact
`‖u‖` elsewhere). Both problems are written out in [FORMULATION.md](FORMULATION.md).

The model is built with ExaModels and solved by MadNLP with **exact Hessians**, the monotone barrier rule,
`SparseKKTSystem` + MUMPS and exact equalities, in one of two execution modes:

| Mode | Model evaluation | MadNLP and linear solver |
|---|---|---|
| `cpu` (default) | ExaModels and the derivative kernels on the CPU threads | CPU |
| `gpu` | the same model on a CUDA device; `HostModel` copies the point there and the values back | CPU |

Both modes solve the same KKT systems; the evaluations agree to rounding (`test/gpu_tests.jl`). The
deterministic reoptimisations always run on the CPU. Convergence means MadNLP's `SOLVE_SUCCEEDED` (and,
reported separately, at most five consecutive restoration iterations).

## Quick start

From the repository root, with Julia 1.12 or newer (validated with 1.13). The deterministic fuel-optimal
references (`output/*_deterministic_traj_opt/fuel_optimal/`) and the Psyche export
(`data/psyche_inputs_fuel_refine2.npz`) must exist.

```sh
# once
julia --project=madnlp_stochastic_traj_opt -e "using Pkg; Pkg.instantiate()"

# step 3 on the CPU: one case, or all five (--case=cr3bp: the three CR3BP cases)
julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl --case=lyapunov_l1_to_l2
julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl --case=all
# step 4, model evaluated on the GPU
julia --project=madnlp_stochastic_traj_opt --threads=4 madnlp_stochastic_traj_opt/run.jl --case=all --objective=chance_sum --mode=gpu

# Monte Carlo validation and figures alone (run.jl does this after every solve)
python madnlp_stochastic_traj_opt/tools/validate.py cpu --case=nrho_l2_to_dro
python madnlp_stochastic_traj_opt/tools/validate_psyche.py madnlp_stochastic_traj_opt/output/gpu/psyche --mode=gpu
```

`run.jl --help` lists the options: `--case`, `--objective`, `--mode`, `--max-iter`, `--quiet`,
`--no-validation`, `--validation-python`. After every solve `run.jl` runs the Monte Carlo validation with
the case's Python validator on the device of the solve (`cpu`: JAX on every CPU core; `gpu`: JAX on CUDA,
after the solve's cached GPU memory is released; a missing GPU is an error, never a CPU fallback). The
validator Python comes from `--validation-python`, `MADNLP_VALIDATION_PYTHON` or `PSYCHE_PYTHON`, else
`~/anaconda3/envs/project/bin/python`, else `python3`, and needs JAX. A failed validation is reported with its
command and never discards the solve. The first run of a session compiles the derivative kernels (included in
the reported times). On an 8 GB laptop GPU, run one GPU process at a time.

## Outputs

Every file of a run goes to `output/<mode>/<case>/` (step 3) or `output/<mode>/<case>_chance_sum/` (step 4);
`MADNLP_STOCHASTIC_OUTPUT_ROOT` replaces `output/`. A new run of the same case, problem and mode replaces it.

| File | Content |
|---|---|
| `<case>.npz` | Solution arrays with the JAX key names where they apply, the reference trajectory, node scales, seed data and the formulation constants `validate.py` replays |
| `<case>_diagnostics.toml` | Solver status, final KKT measures, restoration statistics, independently recomputed residuals, cost shares, settings and timings |
| `<case>_madnlp_nominal.log` | MadNLP log of the fuel reoptimisation |
| `<case>_madnlp_stochastic.log` | MadNLP log of the stochastic solve |
| `validation/` | Monte Carlo (50 000 samples, jax.random seed 42), its JSON summary and the figures |

Psyche writes `psyche_nominal.npz`, `psyche_seed.npz` and `psyche_solution.npz` instead of `<case>.npz`, and its
validation adds the Mars B-plane figure. `tools/summarize_runs.py [cpu|gpu]` tabulates the diagnostics of a
mode into `output/<mode>/summary.csv`.

| Report field | Meaning |
|---|---|
| `madnlp_status`, `solver_success` | MadNLP's termination at its scaled KKT tolerance (1e-6) |
| `restoration_requirement_met` | no run of more than five consecutive restoration iterations |
| `max_constraint_violation` | independently recomputed original residual (diagnostic) |
| `converged` | solver success and the restoration requirement (Psyche also needs a hyperbolic closest-approach minimum) |
| `final_kkt` | scaled and unscaled final KKT measures |

## Results

All runs reach MadNLP's `SOLVE_SUCCEEDED` (CPU runs with 20 Julia threads, GPU runs with 4 on an RTX 5060 Laptop GPU).
Columns: iterations / longest consecutive restoration run / solve time; the objective is the problem's cost at the
solution (CPU and GPU agree to 5 digits or better; the CPU value is shown).

| Case | Step 3 objective | Step 3 cpu | Step 3 gpu | Step 4 objective | Step 4 cpu | Step 4 gpu |
|---|---:|---|---|---:|---|---|
| Lyapunov L1 → L2 | 13.539362 | 233 / 1 / 96 s | 253 / 1 / 129 s | 14.206785 | 341 / 0 / 111 s | 1163 / 113 / 310 s |
| Halo L2 → L1 | 100.116382 | 146 / 0 / 80 s | 146 / 0 / 106 s | 96.365566 | 323 / 0 / 127 s | 324 / 0 / 154 s |
| NRHO L2 → DRO | 103.710839 | 172 / 0 / 105 s | 152 / 0 / 118 s | 99.927013 | 824 / 8 / 433 s | 1835 / 48 / 812 s |
| Earth–Mars | 57.302198 | 396 / 0 / 122 s | 475 / 1 / 136 s | 54.326447 | 1037 / 9 / 269 s | 1122 / 31 / 219 s |
| Psyche | 89.636568 | 432 / 0 / 343 s | 421 / 0 / 212 s | 89.390053 | 1392 / 67 / 729 s | 1310 / 56 / 532 s |

The step 4 runs with a restoration run longer than five iterations report `converged = false` (the report's
restoration requirement), although MadNLP succeeded. Psyche's step 4 starts from the cost-weight seed: from Bryson's
seed it ends at the iteration limit (CPU) or in a false infeasibility certificate (GPU).

## Settings

| Setting | Value |
|---|---|
| tracking weights `w_P`, `w` | Lyapunov 0.01994, 33.56; Halo 0.2128, 118.4; NRHO 0.1531, 129.2; Earth–Mars 0.1530, 151.8; Psyche 0.0102, 54.3 (95 % mean-control share) |
| eigenvalue smoothing `s`, spectral radius floor `f` (step 3) | 1e-6, 1e-6 |
| slack factor floor (step 4) | 5e-4 `T_max` |
| terminal margin factor floor (step 4) | 1e-5 |
| sigma-point mass-flow smoothing δ (feedback-carrying outer points) | 5e-4 |
| MadNLP `tol` = `acceptable_tol` | 1e-6 |
| barrier | monotone, μ from 0.1 down to 1e-10 |
| `alpha_min_frac`, `constr_mult_init_max` | 0.05, 1e4 |
| reoptimisations | `tol` 1e-8, at most 3000 iterations (Psyche: μ₀ = 1e-6, `bound_push` 1e-8) |

The constants are at the top of `src/options.jl` and `src/solver/madnlp.jl`.

## Method

| JAX (`stochastic_traj_opt_jax.py`) | Julia (`src/`) |
|---|---|
| `load_ref_traj` (incl. `np.interp`) | `reference.jl` |
| `reoptimize_reference_trajectory` | `nominal.jl` (fuel, throttle and direction, MadNLP/MUMPS) |
| `Normalization`, navigation covariance, diffusion units | `problem.jl` |
| `integration_map` (Dopri8), `stochastic_integration_map` (XMDS2 RK9) | `dynamics.jl`, tableaux in `generated/` |
| `build_propagation_arc_map` (33 outer, 198 nested sigma points) | `unscented.jl` (literal), `oracles/ut_moments.jl` (kernels) |
| `normalized_arc_jacobians`, `tvlqr_gains`, `build_initial_guess` | `initialization.jl` |
| seeded `TrajectoryNLP` | `model.jl`, `oracles/` |
| `recover_solution` | `diagnostics.jl` (literal arc map, independent of the kernels) |
| `monte_carlo_rollouts`, `compute_diagnostics`, `save_outputs` | reused as is by `tools/validate.py` |

A single-phase case with N arcs has 57N + 7 variables and 37N + 13 constraints in step 3 (step 4: 64N + 28 and
43N + 34); construction asserts both counts. Exact second derivatives:

- **Objective.** Native ExaModels algebra. The nested-UT executed-control covariance has an exact closed form,
  `T = (1/d)[(1+σ₂²−σ₄²) K M Kᵀ + σ₄² tr(K M Kᵀ) I] + Q_G(S) + εI` with `M = d P₆ + d R_nav`; it agrees with the
  literal 198-point sum to about 1e-15.
- **UT moments.** An `ExaModels.add_eval` oracle on the native defect rows (one per set of UT parameters; Psyche
  has two), composed with the Cholesky factor of the predicted covariance (`oracles/ut_factor.jl`). The moment
  Hessian is `Σ_s W_sᵀ [H_s W_s + 2w J_sᵀ N (V_s − V̄)] + ∇²ζ`, built from per-sigma-point forward-over-reverse
  RK9 Hessians (a hand-written discrete adjoint evaluated with dual numbers). The Gauss–Newton term uses the
  centered `V_s − V̄` and the deviations use corrected two-pass means (accurate to 1e-14 on Earth–Mars, 1e-12 on
  Psyche against a 128-bit reference). The state spread is the node factor itself, `S = √d diag(std) L`, so the
  pre-map curvature `∇²ζ` comes only from the executed controls.
- **Thrust chance rows.** `VectorNonlinearOracle`s on the closed-form `T`: the smoothed trigonometric 3×3
  eigenvalue formula (step 3), or the six epigraph rows (step 4).

## Kernels

All cases, Psyche included, use the same KernelAbstractions kernels (threads on the CPU, CUDA on the GPU). The
per-arc kernels are shared; the per-sigma-point derivatives are organized per backend in
`oracles/ut_moments.jl`:

| | CPU | GPU |
|---|---|---|
| Sigma-point derivatives | one thread per sigma point, dual numbers over all 57 arc inputs | stages split over threads: `W_s` in chunks of 8 of the 57 directions, `J_s` and `H_s` per point, `V_s` and `B_s` one column per thread |
| `Σ_s W_sᵀ B_s` | one entry per thread | shared-memory tiles |

On the same device the two layouts give bitwise identical results; the 10 sigma-point directions of the RK9
step are always seeded together (StaticArrays' `dot` sums with `@simd`).

## Verification

```sh
julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/test/runtests.jl        # CPU
julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/test/runtests.jl --gpu                 # + CUDA
```

- `configuration_tests.jl`: command-line parsing, the two problems, the execution modes and the output layout.
- `dynamics_tests.jl`: vector-field VJP, discrete RK adjoint and forward-over-reverse Hessians against central
  differences (both tableaux).
- `model_tests.jl` (3 arcs, both problems): oracle rows against the literal arc map, the objective, the Jacobian
  and the full Lagrangian Hessian against central differences, and the step 4 epigraph rows.
- `psyche_tests.jl`: uncertainty model, ballistic Mars, B-plane Jacobian, and both Psyche models against the
  literal arc map with directional central differences of the Jacobian and Lagrangian Hessian.
- `gpu_tests.jl`, `psyche_gpu_tests.jl`: CUDA evaluations against the CPU model.
- `tools/validate.py`, against the JAX reference: TVLQR gains and gain scales to about 1e-15, propagated seed
  covariances to about 1e-10, and the reference NLP residuals at the Julia optimum.

`tools/benchmark_startup.jl source|package` measures the package's load and first-workload time.

## Layout

```text
run.jl                       command-line entry point
src/CR3BPStochastic.jl       module
  options.jl                 problems, constants, case data, option validation, output directory
  cli.jl                     command-line parsing and the run loop
  generated/                 tableaux and case constants (tools/export_reference_data.py)
  packing.jl problem.jl dynamics.jl unscented.jl
  reference.jl nominal.jl initialization.jl model.jl
  oracles/                   dual-number seeds and kernel launch; UT-moment, factor, chance and slack oracles
  solver/                    MadNLP settings, restoration guard, host view of the device model
  diagnostics.jl solve.jl output.jl validation.jl
  psyche/                    multiphase Psyche case
test/                        runtests.jl (+ --gpu)
tools/
  validate.py validate_psyche.py jax_backend.py   Monte Carlo validation and figures (JAX)
  summarize_runs.py          output/<mode>/summary.csv from the diagnostics
  export_reference_data.py   regenerate src/generated/ from the Python sources
  export_psyche_inputs.py    export the Psyche mission arrays
  benchmark_startup.jl       load and first-workload time
output/cpu/  output/gpu/     the runs of both problems (cpu: CPU evaluation; gpu: GPU evaluation)
```
