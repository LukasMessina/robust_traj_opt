# Validation record

Validated on 2026-09-22 using Julia 1.12.6 and the pinned environment. GPU
checks use an NVIDIA RTX 5060 Laptop GPU. All default runs use uniform
positive `1e-5` gain entries, smoothing `1e-5`, terminal factor floor `1e-7`,
tolerance `1e-8`, and one stochastic solve. A third consecutive restoration
iteration is blocked and rejected.

## CPU line-search prerequisite

Before refactoring, the CPU implementation was run with the sole solver
change `alpha_min_frac=0.05`. It converged in **1700 iterations with zero
restoration iterations**, objective `5.137106176756375`, maximum original
constraint violation `5.195456972625911e-10`, and dual infeasibility
`9.825473767932635e-15`. Verification against the unchanged Python UT and
Diffrax ShARK functions passed. This satisfied the requested prerequisite
for making `0.05` the line-search default in every mode and proceeding with
the refactor. The report is retained in [baselines/cpu.toml](baselines/cpu.toml).

## Refactor validation

The refactored CPU CLI was run with all numerical and postprocessing defaults,
including 8192 Monte Carlo paths, seed 42, and plotting. It converged in 1700
iterations with no restoration. **Every saved solution array was identical**
to the pre-refactor CPU `alpha_min_frac=0.05` result (maximum difference zero).
The original Python objective differed by `8.88e-16`; maximum residual
differed by `1.11e-16` and met the original `1e-8` acceptance threshold.

- 182 CPU tests passed: ShARK/UT kernels, exact derivatives, configuration,
  seed propagation, CLI/output paths, native restoration enforcement, model
  counts, and full-model directional derivative checks. The Lagrangian
  Hessian directional relative error was `4.83e-10`.
- Both closed/open-loop Monte Carlo trajectory arrays have shape
  `(21, 4, 8192)` and finite entries; covariance arrays have shape `(21,4,4)`
  and are positive semidefinite to numerical tolerance.
- All four expected PNG figures were produced at 1280-by-1120 resolution.
- `Project.toml` and `Manifest.toml` are byte-for-byte unchanged.
- The CLI help was checked from both the repository and application directory.

All three refactored solver modes converged with the default
`alpha_min_frac=0.05`, no continuation, and **zero restoration iterations**:

| Mode | Iterations | Objective | Maximum original violation | Dual infeasibility |
|---|---:|---:|---:|---:|
| CPU | 1700 | 5.137106176756375 | 5.195457e-10 | 9.825474e-15 |
| Condensed GPU | 1179 | 5.137106001962187 | 1.711484e-9 | 5.785303e-15 |
| Hybrid GPU | 1745 | 5.137106176756376 | 5.195456e-10 | 3.143319e-15 |

All 23 GPU kernel, complete-transcription, and regularized hybrid linear-system
checks passed, as did 18 assertions across the two full GPU solves and the
CPU/GPU objective comparisons. Both GPU results passed evaluation by the
original Python UT/Diffrax ShARK functions at the `1e-8` acceptance threshold.
Every mode retained 504 model variables and 456 constraints. The GPU solves
ran after the derivative/linear-algebra tests in a shared process; their
iteration counts differ from the earlier isolated experiments, while their
accepted objectives agree to floating-point precision.

Current generated evidence is under `../output/`: CPU/GPU test logs and the
three modes' solution/diagnostic files. Python verification results are saved
beside each checked solution. The maintained commands are in the README.

## Preserved GPU baseline

These reports predate the structural refactor and already use
`alpha_min_frac=0.05` and the current uniform initialization:

| Mode | Iterations | Original constraint violation | Dual infeasibility | Restoration total / longest run |
|---|---:|---:|---:|---:|
| [Condensed GPU](baselines/gpu_condensed.toml) | 698 | 1.711484e-9 | 3.039236e-15 | 2 / 2 |
| [Hybrid GPU](baselines/gpu_hybrid.toml) | 1026 | 5.195456e-10 | 3.649858e-15 | 0 / 0 |

The accepted objectives were `5.137106001962186` and `5.137106176756375`.
The small objective difference accompanies condensed equality relaxation.
Convergence is assessed using solver success, the restoration policy, and
original feasibility, not identical iteration counts across execution
contexts. Earlier experiments showed sensitivity to Julia optimization level
and execution context; a successful run is not a universal convergence claim.

## Numerical decisions retained

The earlier unscaled original Bryson initialization was rejected in both GPU
modes: condensed at iteration 90 and hybrid at 74, each requesting a third
consecutive restoration step. Uniform initialization was selected after the
successful runs above and subsequent CPU verification. Both seeds were
dynamically feasible; their initial chance/terminal constraints could be
violated. The Bryson implementation remains available for explicit experiments.

Earlier equality-Jacobian diagnostics found full row rank at the sampled
initial and intermediate points. Initial smallest singular value was
`5.1132849586e-3`; a converged hybrid diagnostic had minimum singular value
`2.6205600567e-3`. These were different initialization/smoothing experiments
and do not establish rank at every possible iterate. Adding redundant
nonnegative bounds on state Cholesky diagonals did not resolve the observed
hybrid failure, so the original constraint rows were retained without those
extra bounds. The hybrid regularization adapter and its full KKT residual
tests remain in the maintained project.

## Scope of the refactor

The former monolithic script was separated into configuration, dynamics,
initialization, model, solver, diagnostics, orchestration, postprocessing,
and CLI files. The existing module name and Julia API were retained. The
entry point is now `run.jl`; tests and the Python verifier live in `test/`
and `tools/`. Output defaults resolve from the project directory.

Old sweep drivers, duplicate reports, obsolete generated runs, and caches were
removed from the application folder. The compact reports in `baselines/`
preserve the accepted reference runs. The objective, constraints, numerical
kernels, ShARK tableau, unscented transform, original variable count, and
solver adapters were preserved. The only numerical default changed in this
refactor is CPU `alpha_min_frac`, from 0 to the verified common value 0.05.

## Smaller eigenvalue smoothing: 2026-09-23

The normal CLI was tested in each mode with `--eigenvalue-smoothing=1e-6`.
All other numerical defaults were retained: uniform `+1e-5` gain entries,
`alpha_min_frac=0.05`, tolerance `1e-8`, terminal factor floor `1e-7`, maximum
10000 iterations, and at most two consecutive restoration steps. Condensed
GPU retained relaxation `1e-9`; CPU/hybrid retained exact equalities. Each
case used a fresh energy/UT-ShARK seed and one stochastic solve. Only plots
and Monte Carlo postprocessing were disabled. Julia optimization level was 2;
the three mode processes were launched concurrently.

**None of the three runs converged under the current restoration policy.**
Each returned `USER_REQUESTED_STOP` when the guard blocked a request for a
third consecutive robust restoration step; none reached the iteration limit.

| Mode | Iterations | Original constraint violation | Dual infeasibility | Restoration total / longest run |
|---|---:|---:|---:|---:|
| CPU | 510 | 4.312705e-1 | 7.231003 | 2 / 2 |
| Condensed GPU | 2323 | 1.821116e-1 | 2.605271e-3 | 2 / 2 |
| Hybrid GPU | 263 | 1.054184 | 14.974243 | 3 / 2 |

Hybrid's three restoration iterations occurred across separate episodes;
the maximum consecutive count was two. All three final iterates also failed
the original `1e-8` feasibility acceptance threshold. They are diagnostic
iterates, not accepted solutions. Initial mean defects were zero and
covariance defects `1.11e-16` in every case, with 504 variables and 456
constraints throughout. These are measured outcomes for these executions,
not a claim that convergence with `1e-6` is impossible under other settings.
The production default remains the previously validated smoothing `1e-5`.

Raw logs, final iterates, diagnostics, and `summary.csv` are retained under
`../output/smoothing_1e-6_20260923/`. Reproduce from the application directory,
replacing `cpu` with `gpu_condensed` or `gpu_hybrid` as needed:

```sh
julia --project=. run.jl --mode=cpu --eigenvalue-smoothing=1e-6 --no-plots --no-monte-carlo --output=output/smoothing_1e-6_20260923/cpu
```
