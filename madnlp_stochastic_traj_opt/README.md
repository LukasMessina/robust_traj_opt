# MadNLP stochastic trajectory optimization

Julia implementation of `../double_integrator_stochastic_traj_opt.py` using
ExaModels and MadNLP. It retains the original chance-constrained formulation,
25-point augmented unscented transform, and ShARK stochastic integrator,
including Brownian increments and space-time areas. There are **504 decision
variables and 456 constraints** for the default 20-interval problem.

## Run

From the repository root, with Julia 1.12:

```sh
julia --project=madnlp_stochastic_traj_opt -e "using Pkg; Pkg.instantiate()"
julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/run.jl
julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/run.jl --mode=gpu_condensed
julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/run.jl --mode=gpu_hybrid
```

GPU execution requires an NVIDIA CUDA device. CPU mode loads neither CUDA nor
the hybrid adapter. The pinned `Project.toml` and `Manifest.toml` are unchanged
by the refactor. This is a Julia application environment; use `run.jl` or
include the module directly as shown below.

For optimization only, add `--no-plots --no-monte-carlo`. Use `--quiet` to
suppress iteration tables and `--help` for all command-line options.
First execution compiles the model and derivative kernels; that compilation
cost is separate from warm solver runtime.

## Current defaults

| Setting | Value |
|---|---|
| Execution mode | `cpu` |
| Horizon | 20 intervals, `dt=0.25`, final time 5 |
| Initial mean/control trajectory | Deterministic minimum-energy transfer |
| Initial gain matrices | Every entry **positive `1e-5`**, all modes |
| Seed propagation | Full UT–ShARK mean and covariance propagation |
| Eigenvalue smoothing | `1e-5` |
| Terminal factor diagonal floor | `1e-7` |
| Control norm epsilon | `1e-6` |
| Solver KKT tolerance | `1e-8` |
| Original feasibility acceptance | `1e-8` (`10 * feasibility_tol`, with `feasibility_tol=1e-9`) |
| Maximum stochastic iterations | 10,000 |
| Consecutive restoration limit | 2; block and reject a third |
| Line-search `alpha_min_frac` | **`0.05` in all modes** |
| Initial barrier | `0.01` stochastic; `0.1` nominal |
| Condensed GPU relaxation | `1e-9`, one stochastic solve, no continuation |
| Hybrid augmentation | `1e7`, exact equalities |
| Monte Carlo | Enabled, 8,192 scrambled Sobol paths, seed 42 |
| Plots | Enabled, rendered off-screen |

The objective, path constraints, chance probabilities, initial/target
covariances, and original constraint counts are preserved. Details are in
[docs/formulation.md](docs/formulation.md). The hybrid adapter is explained in
[docs/hybrid_kkt.md](docs/hybrid_kkt.md), and measured validation results are in
[docs/validation.md](docs/validation.md).

Optional initialization choices remain available:

```sh
# Original, unscaled Python Bryson TVLQR, in any mode:
julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/run.jl --initial-gain-uniform=none --tvlqr-control-weight-scale=1

# TVLQR rescaled to a maximum absolute initial entry of 1e-5:
julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/run.jl --initial-gain-uniform=none --initial-gain-max-abs=1e-5
```

These options affect initialization only and introduce no bounds on optimized
gains. Explicit uniform initialization and TVLQR rescaling are mutually
exclusive. The optional hybrid TVLQR penalty multiplier remains 1000 unless
overridden; it does not affect uniform initialization. In Julia,
`Options(warm_start_gains=false)` selects a zero-gain seed.

## Julia API

```julia
include("madnlp_stochastic_traj_opt/src/DoubleIntegratorStochastic.jl")
using .DoubleIntegratorStochastic

options = Options(execution_mode=:gpu_hybrid, make_plots=false,
                  run_monte_carlo=false)
result = solve_problem(options)
save_results(result)

result.report
result.solution.means              # 4 × (N+1)
result.solution.feedforward        # 2 × N
result.solution.gains              # 2 × 4 × N
result.solution.state_covariances  # 4 × 4 × (N+1)
```

`solve_problem` returns the result without writing files. `save_results`
writes arrays and diagnostics and performs the requested rollouts/plots.
The CLI performs both and exits with an error if convergence, restoration,
or original feasibility requirements fail; diagnostic output remains saved.

## Layout

```text
run.jl                        Command-line entry point
Project.toml, Manifest.toml    Pinned application environment
src/
  DoubleIntegratorStochastic.jl  Module and include order
  options.jl                    Configuration and validation
  dynamics.jl                   ShARK, UT, diffusion calibration
  initialization.jl             Energy trajectory and gain/covariance seeds
  model.jl                      ExaModels stochastic transcription
  solver.jl                     CPU/GPU backends and solver setup
  restoration_guard.jl          Consecutive restoration enforcement
  regularized_hybrid_kkt.jl      Hybrid dual-regularization adapter
  *_oracle.jl                   Exact sparse UT/chance derivatives
  diagnostics.jl                Independent residual/solution evaluation
  solve.jl                      Solve orchestration and report
  postprocess.jl                Rollouts, plots, NPZ/TOML output
  cli.jl                        Command-line parsing
test/                         CPU/GPU numerical regression tests
tools/verify_python_reference.py  Original Python/Diffrax verification
docs/                         Method, validation, compact baseline reports
output/                       Generated results (ignored by Git)
```

The previous `julia_double_integrator` directory and its root-level script
were replaced by this layout. Update old launch commands to `run.jl` and
old includes to `src/DoubleIntegratorStochastic.jl`. The module name and
`Options` / `solve_problem` API are retained.

## Tests

From the repository root:

```sh
# CPU kernels, configuration, restoration guard, transcription and derivatives:
julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/test/runtests.jl --model

# Also run a complete CPU optimization:
julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/test/runtests.jl --solve

# GPU kernels, transcription, hybrid linear algebra and both solver modes:
julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/test/runtests_gpu.jl

# GPU derivative/linear-algebra checks without complete optimizations:
julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/test/runtests_gpu.jl --derivatives-only
```

The Python verifier requires NumPy, SciPy, JAX and Diffrax, but not SNOPT. It
extracts the reference functions from the unchanged Python file next to this
project and evaluates them at a saved Julia solution:

```sh
python madnlp_stochastic_traj_opt/tools/verify_python_reference.py madnlp_stochastic_traj_opt/output/cpu
```

## Results

Output defaults to `madnlp_stochastic_traj_opt/output/<mode>/`, independent of
the launch directory. An explicit `--output=PATH` is relative to the caller's
working directory unless absolute.

- `solutions.npz`: trajectory, gains, covariances, factors, calibrated diffusion.
- `diagnostics.toml`: convergence, original residuals, restoration, settings, timing.
- `monte_carlo.npz`: closed/open-loop trajectories and empirical covariances.
- Four PNG figures when both plotting and rollouts are enabled.

NPZ axis conventions match the Python reference. Monte Carlo streams are
reproducible within Julia; they are not bitwise identical to NumPy streams.
Obsolete sweep drivers, caches and historical generated runs are excluded from
the maintained project. Compact numerical evidence is retained in
`docs/baselines/`.
