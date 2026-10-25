# Deterministic trajectory optimisation

`deterministic_traj_opt.py` is the entry point for all five transfers and both objectives. The implementation is split between physical models, transcriptions, integration, refinement, diagnostics and plotting in `deterministic_optimization/`. Psyche mission-data reconstruction is separate in [psyche_mission_events_reconstruction/README.md](psyche_mission_events_reconstruction/README.md).

## Run

From the repository root:

```powershell
conda activate det_traj_opt_env
python deterministic_traj_opt.py --case all --objective both --jobs 3
```

For individual runs, `--case` accepts `nrho_l2_to_dro`, `halo_l2_to_halo_l1`, `lyapunov_l1_to_l2`, `earth_mars`, `psyche`, `cr3bp`, or `all`; `--objective` accepts `fuel`, `energy`, or `both`. The default is all ten cases, sequentially (`--jobs 1`). Results go into the three case-family folders below, each with `fuel_optimal` and `energy_optimal` subfolders. A successful run replaces the selected case's published files. Failed runs never overwrite accepted results. Jobs are independent processes with one BLAS thread per solver. There is no solver runtime cap.

To rerun Psyche only, at its current local-defect target of **5e-14**:

```powershell
python deterministic_traj_opt.py --case psyche --objective both --jobs 2
python -m unittest discover -s tests
```

To regenerate figures without solving:

```powershell
python deterministic_traj_opt.py --case all --objective both --plots-only
```

Install the environment from `environment-deterministic.yml` if needed. The checked environment uses Python 3.12, CasADi 3.7.2 and the bundled IPOPT/MUMPS solver, with TudatPy and SpiceyPy for reconstruction. Plotting calls the unchanged `plotter.py` utilities: Computer Modern/serif, LaTeX, original colours, dimensions, line/legend conventions, and 240-dpi PNG exports. A working LaTeX installation and `dvipng` are required. Its cache needs normal filesystem access. Matplotlib's cache defaults to `.cache/matplotlib` in the workspace.

## Current configuration

| Setting | Three CR3BP transfers | Earth–Mars | Psyche |
|---|---|---|---|
| Initial arcs, each objective | 50 | 50 | 60 / 50 / 90 / 50 |
| Shooting RK7 uniform substeps/arc | 2 | 20 | 20 |
| Defect RK7 uniform substeps/arc | 64 | 64 | 64 |
| Relative defect target | 5e-13 | 1e-13 | 5e-14 |
| Maximum refinement passes | 8 | 8 | 8 |
| Maximum total arcs | 1800 | 1800 | 1800 |

Both HS and shooting use:

```ini
mu_strategy = adaptive
tol = 1e-9
constr_viol_tol = 1e-10
max_iter = 5000
acceptable_iter = 0
bound_relax_factor = 0.0
bound_push = 1e-8
bound_frac = 1e-8
hessian_approximation = exact
nlp_scaling_method = gradient-based
print_level = 5
sb = yes
```

No explicit `compl_inf_tol`, `mu_min` or `mu_target` override is retained in these reference runs. CasADi expands the Psyche HS model; it leaves shooting and the single-phase models unexpanded, as in the reference implementation. Exact first/second derivatives include state, control, duration and time-dependent frame translations.

## Mathematical formulation and workflow

Every objective starts independently from a fresh dynamics rollout with **1e-4 N per thrust component**, integrated with **two uniform RK7 substeps per initial arc**. Reference outputs, spacecraft ephemeris trajectories and the other objective's solution never initialise a solve.

The workflow is **rollout → same-objective HS once → multiple shooting with h-refinement**. Controls are constant on each arc, with throttle and unit direction:

\[
\mathbf T=T_{max}\alpha\mathbf d,\quad 0\leq\alpha\leq1,\quad
\mathbf d^T\mathbf d=1,\quad \dot m=-T_{max}\alpha/(I_{sp}g_0).
\]

There is no epigraph slack or smoothed thrust norm in this formulation. Stored legacy `sigma`/magnitude fields represent the derived `Tmax*alpha`. Unit-direction residuals can cause tiny differences from the actual thrust-vector norm; these are reported, not used as extra acceptance gates. Directions are unconstrained in their physical effect at zero throttle, which can worsen conditioning.

The decision variables are seven state components at each mesh boundary, one throttle and a three-component unit vector per arc, plus Psyche's two free planet-centred durations. Other case durations and boundary states remain fixed. For N arcs, a single-phase problem has `7*(N+1) + 4*N` primal variables. Psyche has the sum over its four phases plus two duration variables. HS midpoints are algebraic expressions, not additional decision states.

Fuel is initial minus final mass [kg], scaled by wet mass in the NLP. Energy is **0.5 ∫ dot(T,T) dt** [N²s], scaled by `Tmax² * total transfer duration` in the NLP. The energy uses the physical vector even away from the unit-direction constraint. Do not compare these energy values with an unhalved squared-thrust integral.

HS must converge before shooting starts. The first shooting stage receives the same HS states, throttle, direction, mesh and, for Psyche, exit times. Once a shooting NLP converges, 64-substep RK7 propagates each arc independently from its optimised start and compares with its optimised endpoint. Acceleration uses the physical thrust vector and mass flow uses `Tmax*alpha`, matching the NLP.

The local criterion is the maximum of position-vector, velocity-vector and mass errors, divided separately by the maximum corresponding endpoint magnitude, with a **1e-13 nondimensional denominator floor**. Only failing arcs are bisected. If the arc cap limits capacity, the largest defects are selected first. Existing nodes and controls are retained. CR3BP/Earth–Mars initialise new states with the existing cubic Hermite interpolation; Psyche evaluates the preceding shooting arc by partial RK7 propagation. Child arcs inherit their parent's throttle and direction. No control polishing or new integration split is introduced.

Acceptance requires **NLP convergence and all local defects meeting their target**. Refinement ends successfully when those pass; otherwise it stops with an explicit incomplete status on NLP failure or either mesh limit. Global reintegration and thrust/mass-flow discrepancies are diagnostics only.

CR3BP retains Moon-surface exclusion at **mesh boundaries only**, with Moon radius 1737.5 km, in both HS and shooting. No additional interior Moon checks enter the NLP. Earth–Mars retains its existing constraints; no new surface inequalities have been added. Psyche retains central-body surface checks and its two exit-boundary equalities/outward inequalities, with no inside/outside-SOI path inequalities. See its README for all physical assumptions and event data.

## Outputs and propagation diagnostics

Published folders contain only case-prefixed PNG plots and one CSV, JSON and NPZ per case/objective. The JSON includes the configuration, solver/refinement history, convergence status, units and propagation diagnostics. NPZ common fields (`mesh_fraction`, `x`, `u`, `t_dense_days`, `x_dense`, `u_dense`, `history`, etc.) and CSV columns retain the original format and nondimensional state units; thrust columns remain in N. `sigma` is now derived from throttle, not an epigraph decision variable. Radau stage fields are empty and interval degrees are zero. Psyche's common arrays use Sun-centred J2000 and the solar phase's scales; exact planet-relative SI phase arrays and free exit times are additionally retained. The stochastic reference loader remains compatible and now reads the current multiple-shooting solutions.

Solver logs, checkpoints, source snapshots and intermediate stages are kept outside the published folders in a timestamped `.cache/deterministic_runs/` work directory. `--work-dir PATH` selects a new work directory. `--output ROOT` changes the parent of the three case-family folders; it no longer denotes a flat diagnostic batch. `--export-from BATCH` publishes existing detailed results without reoptimising. A `converged` result always means both NLP and local-defect acceptance.

Public figures match the original 3D, projections (XY for the planar Lyapunov case), thrust, interval defect and three-panel verification plots. CR3BP also retains Moon clearance. Psyche adds textured Earth-departure and Mars-flyby close-ups with thrust arrows. Verification plots include position, velocity and mass errors: single-phase cases use the tightest DOP853 settings below; Psyche uses the standard DOP853 history. Tighter/tightest comparisons remain in the JSON. Solver/refinement histories remain available in the JSON without extra public plots.

Continuous propagation begins at the fixed initial state, carries state through the whole transfer, and never resets it at shooting nodes. Psyche includes origin translations without resetting to the fixed Mars anchor. Actual thrust norm supplies physical mass flow in the principal diagnostic. Psyche also reports the throttle-law comparison.

* CR3BP/Earth–Mars use DOP853 in nondimensional variables. Tighter/tightest rtol are 1e-13/3e-14. SI position absolute tolerances are 1e-7/1e-8 m; velocity and mass tolerances are 1e-11/1e-12 in m/s and kg, converted to state scales. Maximum steps are arc duration/16 and /32.
* Psyche integrates in SI. Standard/tighter/tightest rtol are 2e-12/1e-13/3e-14. Position atol are 1e-5/1e-7/1e-8 m, and velocity/mass atol are 1e-9/1e-11/1e-12. Base maximum step is min(arc duration/8, 86400 s in solar phases or 900 s in planet-centred phases), multiplied by 1, 0.5 or 0.25.

Reference-integrator differences are sensitivity estimates, not rigorous error bounds. Psyche's multi-year propagation remains sensitive at approximately kilometre scale despite tiny local defects. Surface constraints sampled at finite nodes cannot certify all-times collision avoidance. NLP convergence does not prove a global optimum or physical mission accuracy.

## Validated results and reference comparison

The pre-refactor baseline was frozen and compared against all ten consolidated runs before cleanup. At the user's request, the superseded reference output folders `output/deterministic_multiple_shooting/` and `output/psyche/`, and the `maintenance/` audit tools and archives, were subsequently removed. They were not runtime dependencies. The saved comparison below is a historical record; rerunning the old pairwise comparison requires the removed archives and comparison tool.

The consolidation batch in `output/deterministic/validated/` reproduced the reference initial guesses, final numerical arrays, refined initial guesses, settings, boundaries, objectives, meshes, iteration counts, defects and continuous diagnostics. [reference_comparison.json](output/deterministic/validated/reference_comparison.json) records that historical comparison, before Psyche's target changed from 1e-13 to 5e-14. The following table is that historical baseline. Current accepted results are the case-family JSON/NPZ exports. Parallel runtime differences are not a controlled performance benchmark.

| Case | Fuel-optimal propellant [kg] | Energy-optimal cost [N²s] | Final fuel / energy arcs |
|---|---:|---:|---|
| NRHO L2 → DRO | 22.613366277876 | 71961.984868985 | 98 / 88 |
| Halo L2 → Halo L1 | 26.073498220631 | 93873.706094125 | 77 / 146 |
| Lyapunov L1 → L2 | 2.524385694941 | 4071.350873322 | 64 / 65 |
| Earth–Mars | 396.388173450835 | 1423266.864908242 | 50 / 50 |
| Psyche | 731.116528664271 | 609809.304671808 | 61/50/94/50 / 62/50/92/50 |

The fuel consumed by an energy-optimal solution is a different reported quantity; see the batch report for both quantities for every run.

The CR3BP and Earth–Mars public folders now hold the current multiple-shooting results. `stochastic_traj_opt_jax.py` reads their compatible energy NPZ exports. They are not warm starts for deterministic solves. Original model exports from `deterministic_traj_opt.py` remain available to stochastic consumers; changing the deterministic driver's initial mesh to 50 does not change a stochastic case's historical `nodes` attribute.

## Structure

```text
deterministic_traj_opt.py                  unified CLI and compatible model exports
deterministic_optimization/
  cli.py, settings.py                      orchestration and defaults
  models.py                               CR3BP / Earth–Mars physics
  single_phase.py, psyche.py               HS / shooting formulations
  psyche_driver.py                        four-phase solve/refinement loop
  flows.py, rk.py                          uniform RK7
  psyche_validation.py, diagnostics.py     local defects, acceptance and diagnostics
  derivatives.py, global_reference.py      independent numerical checks
  checkpoint.py, io.py                     result/checkpoint provenance
  plotting.py, reporting.py                original-style figures and reports
  exports.py, psyche_figures.py            compact public files and textured passage plots
  plot_assets.py                          explicit cartographic input downloader
psyche_mission_events_reconstruction/      mission inputs, README and evidence
output/cr3bp_deterministic_traj_opt/{fuel,energy}_optimal/
output/earth_mars_deterministic_traj_opt/{fuel,energy}_optimal/
output/psyche_mission_deterministic_traj_opt/{fuel,energy}_optimal/
.cache/deterministic_runs/                 detailed solver work directories
data/planet_textures/                      maps, orientation kernels and source manifest
```

Active deterministic code has no Radau solve or Radau h-refinement path. Reusable RK components in `integrator.py` remain for other project consumers. Required mission inputs and reconstruction evidence remain in `psyche_mission_events_reconstruction/`.

## Planetary passage figures

Download plotting assets once with `python -m deterministic_optimization.plot_assets`. The source URLs, sizes and SHA256 hashes are saved in `data/planet_textures/manifest.json`; plotting performs no network requests. Earth uses NASA's [Blue Marble October map](https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/base-map/) at 5400×2700, mapped by WGS84 geodetic latitude onto its ellipsoid. Mars uses the [NASA Viking/USGS mosaic hosted by ASU](https://mars.asu.edu/data/mdim_color/), assembled from eight 5760-pixel tiles into an 8192×4096 map, with planetocentric latitude on a 3396.19/3376.20 km ellipsoid. Both maps run from -180° to +180° east, north to south; no longitude mirroring is used.

SPICE rotates Earth from ITRF93 to J2000 at TIP, using the pinned `earth_000101_270119_261023.bpc` (coverage 2000-01-01 to 2027-01-19). Mars uses IAU_MARS from `pck00011.tpc` at the optimized closest approach. Each JSON records the rotation matrix and epoch. These are static surface snapshots overlaid with a time-varying inertial trajectory, not ground tracks or mission-date satellite photographs. The map products are visual context, not precision terrain or spacecraft attitude data. SPICE queries occur only after solving for these figures. Thrust arrows preserve inertial direction and relative magnitude. The heliocentric display uses 0.48 AU at full throttle; passage figures use 1.20 constraint-sphere radii at the largest displayed thrust. Only the passage arrows retain doubled lengths and shaft widths; their arrowheads scale with them. The main 3D and projection arrows use their previous length and thickness. These display scales and the surface epoch remain in the JSON. Passage plots use the main 3D camera (azimuth -60 degrees, elevation 30 degrees), dashed projections onto the box planes, and a window 1.75 times as wide as the preceding passage views (3.381 times the original close-up half-width), resampling a longer trajectory segment; titles and bottom captions are omitted. Visual ellipsoids do not change the optimisation's spherical surface constraints.

## Psyche rerun at 5e-14

Both objectives converged from independent fresh rollouts, starting at **60 / 50 / 90 / 50 arcs**, through same-objective HS and adaptive multiple shooting. The new initial-guess arrays are identical to the consolidation reference; the only optimisation-setting difference is the requested defect target. The preceding 1e-14 batch was stopped at the user's request. Detailed work is in `.cache/deterministic_runs/psyche_tol5em14/`; the accepted results are in the public case-family folders.

| Quantity | Fuel objective | Energy objective |
|---|---:|---:|
| Converged objective | 731.115382349 kg | 609809.272913296 N² s |
| Propellant consumed [kg] | 731.115382349 | 814.520099493 |
| Final arcs | 62 / 50 / 104 / 50 | 80 / 50 / 116 / 50 |
| Refinements after initial shooting mesh | 2 | 5 |
| Maximum relative local defect | 3.713601392e-14 | 4.308869913e-14 |
| Sum of NLP solve wall times [s] | 424.464 | 579.088 |
| Objective change from 1e-13 consolidation reference | -0.001146315 kg | -0.031758512 N² s |

Continuous propagation remains a diagnostic. The endpoint discrepancies below use physical thrust norm for mass flow and carry the state through all phases without resetting it. The published verification figures use the standard history. These estimates remain sensitive to the reference integrator; the tightest result is not a certified truth trajectory.

| Final propagation diagnostic | Fuel | Energy |
|---|---:|---:|
| Standard DOP853 position discrepancy [m] | 17237.329 | 233.647 |
| Standard DOP853 velocity discrepancy [m/s] | 7.034955e-4 | 9.628512e-6 |
| Tightest DOP853 position discrepancy [m] | 652.867 | 737.182 |
| Tightest DOP853 velocity discrepancy [m/s] | 2.664840e-5 | 3.037188e-5 |
| Tighter versus tightest position difference [m] | 780.787 | 629.338 |

Both public NPZ files preserve every native phase mesh/state/control array and exit duration exactly. CSV columns and common NPZ fields retain compatibility. Each Psyche objective directory contains exactly seven PNGs plus CSV, JSON and NPZ. The eight other cases were re-exported without new optimisation; their mesh/state/control arrays still match the validated reference, and the unchanged stochastic readers load the four energy references. All 58 tests pass. Revised 3D, projection, passage and diagnostic figures were visually inspected.
