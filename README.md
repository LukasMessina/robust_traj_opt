# Trajectory optimization and verification

Deterministic and stochastic low-thrust trajectory tools for Earth–Moon CR3BP transfers, Earth–Mars and the three-phase Psyche mission, plus a planar double-integrator example. Run the commands below from this repository's root.

## Repository layout

```text
Code/
├── data/                                    # Verified inputs, ephemerides and provenance
├── deterministic_trajectory_optimization/   # Python/CasADi transcriptions and drivers
├── stochastic_trajectory_optimization/      # Julia/MadNLP and double-integrator SNOPT
├── integrator/                              # RK7, Dopri8, RK9, ShARK and event propagation
├── plotter/                                 # Shared style, figures and cartographic assets
├── verification/                            # Tests, validation, benchmarks and reconstruction
├── environment/                             # Python/Conda and Julia installation environments
├── output/                                  # Simulation results and verification reports
├── solvers/                                 # Local licensed solver installations, ignored by Git
├── README.md
└── .gitignore
```

The layout reduces Python/Julia source files from **123 to 104**. Related I/O, checkpoint and worker-logging routines share `deterministic_trajectory_optimization/runtime.py`; exports and summaries share `results.py`. Numerical integration and plotting have dedicated modules instead of being embedded in solver scripts. Julia's solver adapters, derivative oracles and Psyche implementation are consolidated by responsibility. Mission kernel manifests, downloads and coverage checks share one reconstruction module. `verification/export.py` provides both native input export commands.

`data/mission.py` and `data/ephemeris.py` load validated input data without running SPICE during optimization. Historical provenance paths resolve to the relocated inputs without rewriting the scientific arrays. Reconstruction code lives under `verification/reconstruction/`; its retained inputs and evidence live under `data/psyche/`.

Julia retains `stochastic_trajectory_optimization/Project.toml` as package metadata so normal package loading and precompilation work. The active environment and dependency lock are `environment/Project.toml` and `environment/Manifest.toml`, which reference that local package. This is the sole environment-related metadata outside `environment/`.

## Installation

Tested platform: Linux x86_64 / WSL, Python 3.11 and Julia 1.13.0. CPU runtime dependencies are pinned in `environment/requirements.txt`:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r environment/requirements.txt
python -m pip check
```

This supports optimization and validation from existing mission inputs. Reconstruction and the complete Python test suite additionally require TudatPy. Use the explicit Linux package lock, followed by the pinned Python requirements:

```sh
conda create -n trajectory --file environment/environment-linux-64.lock
conda activate trajectory
python -m pip install -r environment/requirements.txt
# Portable specification, with dependency resolution:
# conda env create -f environment/environment.yml
```

The tested versions include NumPy 2.4.6, SciPy 1.17.1, CasADi 3.8.0, JAX/JAXlib 0.10.2, Diffrax 0.7.2 and TudatPy 1.0.0.dev75. Keep the Tudat/NumPy ABI compatible; see [Tudat installation](https://docs.tudat.space/en/latest/getting-started/installation.html).

```sh
julia --project=environment -e 'using Pkg; Pkg.instantiate()'
```

The relocated Julia environment preserves the existing dependency versions. Use the manifest rather than updating packages when reproducing a run.

Optional NVIDIA acceleration:

```sh
python -m pip install 'jax[cuda12]==0.10.2'
JAX_PLATFORMS=cuda python -c 'import jax; print(jax.devices())'
```

GPU checks used JAX CUDA12 plugin/PJRT 0.10.2, CUDA 12.9 and cuDNN 9.10.2 on an RTX 5060 Laptop GPU with 8 GB. The optional pip CUDA extra is not a separately validated installation lock; consult [JAX installation](https://docs.jax.dev/en/latest/installation.html). Explicit GPU requests fail if CUDA is unavailable. JAX validation disables memory preallocation. Julia evaluates model kernels on the GPU but retains the CPU sparse KKT system and MUMPS linear solve.

SNOPT is optional and used only by the double integrator. Supply a licensed SNOPT-linked pyOptSparse installation (tested: 2.16.0). The launcher defaults to `solvers/snopt/libsnopt_helpers_units.so` and `solvers/snopt/snopt7.lic`; override with `SNOPT_HELPER_LIBRARY`, `SNOPT_LICENSE` and `SNOPT_PYTHON`. Licenses and binaries remain outside version control.

For headless figures, set `MPLBACKEND=Agg` and, if needed, a writable `MPLCONFIGDIR`. Figures use LaTeX/dvipng when available and otherwise fall back to Matplotlib mathtext.

## Deterministic optimization

```sh
python -m deterministic_trajectory_optimization --case all --objective both --jobs 3
python -m deterministic_trajectory_optimization --case psyche --objective fuel --jobs 1
python -m deterministic_trajectory_optimization --help
```

Cases: `lyapunov_l1_to_l2`, `halo_l2_to_halo_l1`, `nrho_l2_to_dro`, `earth_mars`, `psyche`; `cr3bp` selects the first three. Fuel and energy references start independently. The unchanged workflow uses a dynamics rollout, Hermite–Simpson collocation and adaptive RK7 multiple shooting. Throttle and unit direction parameterize thrust. Energy is `0.5 * integral(T' T dt)` in N² s.

Existing references retain their filenames under `output/{cr3bp,earth_mars,psyche_mission}_deterministic_traj_opt/{fuel,energy}_optimal/`. They are inputs to stochastic optimization and must not be deleted as caches. `--output` selects another result root; `--work-dir` selects the detailed solver workspace (default: a timestamped directory under `output/workspaces/`). Use `--plots-only` to regenerate figures and `--export-from` to export a completed workspace.

## Stochastic optimization

Prepare deterministic fuel references first. Export Psyche's fuel mesh for the native solver:

```sh
python -m verification.export psyche --reference output/psyche_mission_deterministic_traj_opt/fuel_optimal/psyche.npz --mesh-refinement 2 --output data/stochastic/psyche_inputs_fuel_refine2.npz
# Regenerate Julia case constants and integration tableaux if their Python sources change:
python -m verification.export tables
```

Exports retain a SHA256 link to the deterministic reference. Regenerate them when that reference changes. Single-phase native cases load their fuel references directly, splitting each interval in two.

```sh
export MADNLP_VALIDATION_PYTHON="$(command -v python)"
# Step 3
julia --project=environment --threads=4 stochastic_trajectory_optimization/run.jl --case=all
# Step 4, GPU model evaluation
julia --project=environment --threads=4 stochastic_trajectory_optimization/run.jl --case=all --objective=chance_sum --mode=gpu
julia --project=environment stochastic_trajectory_optimization/run.jl --help
```

Results are under `output/stochastic/<cpu|gpu>/<case>/`, with `_chance_sum` appended for Step 4. `MADNLP_STOCHASTIC_OUTPUT_ROOT` overrides this root. Repeating a case replaces its run directory's results. Saved reports distinguish solver status, scaled KKT errors and independent physical residuals.

### Preserved numerical formulation

Controls are `u = Gamma*d`, where `Gamma` is throttle and `d` a unit direction. Feedback observes position and velocity. The nested unscented transform includes initial uncertainty, navigation error, acceleration diffusion and proportional execution error. RK9 retains its fixed substeps and prescribed Wiener increments. Mass-flow smoothing `5e-4` applies only to feedback-carrying outer sigma points.

At node `k`, `s_k = sqrt(max(diag(P_seed,k), 1e-12))` and `P_k = diag(s_k) L_k L_k' diag(s_k)`. The initial covariance is fixed. Matching uses lower-factor defects:

```text
L_(k+1) - chol(diag(1/s_(k+1)) P_predicted,k diag(1/s_(k+1))) = 0
```

Arc weights are `N*dt_k/sum(dt)`. Step 3 minimizes mean thrust plus weighted state/control covariance and terminal covariance cost; Step 4 minimizes the thrust chance epigraph and imposes the terminal covariance bound. Both retain exact Hessians, monotone barrier, `SparseKKTSystem`, MUMPS, exact equalities, no bound relaxation and stochastic tolerance `1e-6`. Nominal reoptimization uses `1e-8`; iteration budgets remain 5000/10000 for Steps 3/4.

The Step 4 chance-slack factor floor is **`1e-4` for every case**; the terminal-margin factor floor remains **`1e-5`**. Psyche retains its default cost weights **`w_P=0.0102`, `w=54.3`**, including its TVLQR seed. Other Step 4 cases retain unit Bryson seed weights.

Psyche is Sun–Mars–Sun with a strictly ballistic Mars phase: zero thrust and feedback, conserved mass and continuous covariance. Its closest-approach node, moving-axis B-plane chance constraint, gravitational focusing, event projection, 500 km clearance and risk 0.001 are unchanged. Initial position standard deviation is 2500 km; terminal targets are 1500 km and 2 m/s. The mission starts at reconstructed Earth SOI exit and rendezvous with the asteroid center on **2029-08-01 12:00 TDB**. The 2747 kg initial mass remains a modeling assumption.

## Validation and double integrator

```sh
python -m verification cpu --case lyapunov_l1_to_l2 --samples 50000
python -m verification gpu --case psyche --objective chance_sum --samples 50000
python -m verification cpu --case earth_mars --directory /path/to/run --no-plots
python -m verification.summarize cpu
SNOPT_PYTHON=/path/to/snopt/python bash stochastic_trajectory_optimization/run_snopt.sh -m stochastic_trajectory_optimization.double_integrator
```

Validation replays saved weights, covariance scaling, smoothing and terminal rules. Monte Carlo defaults to seed 42 and 50,000 samples. Single-phase propagation uses three RK9 substeps unless overridden; Psyche uses at least 16 and sample-specific SOI events. The latter intentionally checks a hybrid model different from scheduled NLP frame switches. Small ensembles do not establish rare-event probabilities.

For Step 4, the independent **smoothed** spectral-radius diagnostic differs from the optimized epigraph and the unsmoothed physical chance bound. Older NPZ exports do not contain the epigraph slack variables; compare the Julia model objective and original constraint reports too. Validation does not optimize or alter solver acceptance.

The double integrator keeps its physical model, ShARK integration, Sobol ensembles, regularized control norm, costs, constraints and SNOPT settings. It uses throttle/unit-direction controls and the same seed-scaled square-root covariance strategy as Steps 3/4. Its nominal throttle remains unbounded above; stochastic throttle is bounded by one. A feasible accuracy-limit candidate may restart once with identical settings and must then satisfy optimality and feasibility. Results go to `output/double_integrator/`.

## Mission data and verification commands

```sh
python -m verification.reconstruction.kernels              # Download missing pinned kernels
python -m verification.reconstruction                      # Verify retained mission inputs
python -m verification.reconstruction --output output/mission_inputs_new --rebuild-cache
python -m plotter.assets

JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 python -m unittest discover -s verification -p 'test_*.py' -v
JAX_PLATFORMS=cpu python verification/check_psyche_ballistic_jax.py
julia --project=environment --threads=4 verification/native/runtests.jl --gpu
JAX_PLATFORMS=cpu python -m verification.benchmark /path/to/lyapunov_l1_to_l2.npz --output output/linearization-benchmark.json
```

Reconstruction requires pinned SPICE kernels and writes new inputs only when an unused `--output` directory is supplied. `PSYCHE_RECONSTRUCTION_DATA` selects another input set. Mission verification writes to `output/verification/mission.json`. States are geometric J2000, metres and metres/second, with explicit body-centered origins; ET is TDB seconds since J2000. Kernel identity, coverage, ordering and checksums are enforced. SPICE pools are process-global: use isolated processes rather than threaded SPICE queries.

Set Julia `--threads`, JAX `JAX_NUM_CPU_DEVICES` and BLAS/OpenMP limits deliberately to avoid oversubscription. CPU validation otherwise respects process affinity. Run one large GPU workload at a time on an 8 GB device. The benchmark warms both sequential and batched linearization, synchronizes results, includes transfers and checks numerical agreement; it does not measure whole-solver speedup.

## Reorganization verification

The baseline is the working source immediately before this directory reorganization, including the previous throttle/direction and covariance-factor changes. The mathematical algorithms, coefficients, random seeds, solver settings and arithmetic order were retained. The final report and compared numerical arrays are saved in `output/verification/layout/`.

| Check | Result |
| --- | --- |
| Python suite before and after | 66 passed, 1 Windows-only test skipped in each run |
| Julia CPU/GPU suite | 194 checks passed, including Psyche and both single-phase formulations |
| Double-integrator rerun | All 6 numerical arrays bit-for-bit equal; objective `5.1322474531088`, maximum violation `5.81e-13` unchanged |
| Lyapunov Step 3 rerun, 128 arcs | All 50 numerical arrays equal; objective `13.539362117209283`, 233 iterations and residuals unchanged |
| Lyapunov Step 4 rerun, 128 arcs | All 50 numerical arrays equal; epigraph objective `13.971550811334582`, 268 iterations and residuals unchanged |
| JAX propagation, four cases | All 12 arrays bit-for-bit equal |
| Monte Carlo, 50,000 samples / seed 42 | All 15 numerical arrays and statistics bit-for-bit equal |
| Psyche native input export | All 43 arrays bit-for-bit equal |
| Mission and saved Psyche validation | Three pinned event states reproduced exactly; saved objective difference `8.53e-14` |
| Installation and commands | Julia environment resolved/instantiated without solver dependency changes; `pip check`, exports, summaries and figure generation passed |

Step 4's unchanged smoothed diagnostic objective is `14.224125920325696`, with a reported chance-bound excess of `2.94e-4`; the scaled NLP constraint violation is `9.75e-7`. These retain their original meanings and are not silently reclassified as exact feasibility.

Full reruns covered the double integrator and Lyapunov Steps 3/4. The other cases were covered by numerical tests, saved-solution checks and input verification, not fresh full-mission optimizations. No new performance improvement is claimed for this structural refactor. The previously batched JAX implementation is retained.

See the [machine-readable comparison report](output/verification/layout/report.json) and adjacent logs/arrays for the evidence.
