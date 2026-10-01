# Configuration. Problem data are the constants and `Options` defaults of
# cr3bp_stochastic_traj_opt_jax.py; the solver settings are MadNLP's.

const ACCELERATION_DIFFUSION_KM_S32 = 1e-10
const GATES_PROPORTIONAL_MAGNITUDE_STD = 8.0e-3
const GATES_PROPORTIONAL_POINTING_STD = 9.5993e-3
const INITIAL_POSITION_STD_KM = 50.0
const INITIAL_VELOCITY_STD_KM_S = 1.0e-3
const FINAL_POSITION_STD_KM = 10.0
const FINAL_VELOCITY_STD_KM_S = 1.0e-4
const INITIAL_MASS_STD = 1e-3   # [kg]
const MASS_SCALE = 1e-3         # [kg]

# UT dimension: state, navigation error and the Wiener increments of the single RK9 step per arc.
const UT_DIMENSION = NX + NP + NW
# The reference hard-codes p = sqrt(p2/6 + 1e-30) in its 3x3 eigenvalue formula;
# `eigenvalue_smoothing = 1e-15` reproduces that constant bitwise.
const JAX_EIGENVALUE_SMOOTHING = 1e-15
# SNOPT major feasibility tolerance: the reference's acceptance threshold.
const JAX_FEASIBILITY_TOL = 1e-8
# Required bound on consecutive restoration iterations, checked for every run.
const MAX_RESTORATION_RUN = 5

const REFERENCE_DIR = joinpath(REPOSITORY_ROOT, "output", "cr3bp_deterministic_traj_opt", "energy_optimal")
const OUTPUT_ROOT = joinpath(PROJECT_ROOT, "output")
const CASE_IDS = ("lyapunov_l1_to_l2", "nrho_l2_to_dro", "halo_l2_to_halo_l1")
# :cpu  sparse KKT system + MUMPS, exact equalities
# :gpu  condensed KKT system + cuDSS on a CUDA device, equalities relaxed by `condensed_relaxation`
const EXECUTION_MODES = (:cpu, :gpu)

"""
    Options(; kwargs...)

Problem data default to the JAX reference, except these robustification settings
(reference values in brackets), which are the same in both execution modes:

* `eigenvalue_smoothing = 1e-6` [1e-15]: `s` in `p = sqrt(p2/6 + s^2)` of the smoothed
  largest-eigenvalue formula of the thrust chance constraint;
* `spectral_radius_floor = 1e-6` [1e-7]: `f` in `sqrt(lambda_max + f^2)`;
* `terminal_margin_floor = 1e-5` [1e-4]: lower bound of the terminal margin factor diagonal;
* `mass_flow_smoothing = control_norm_eps` [0]: `delta` in the sigma-point mass flow
  `T_max sqrt(||e||^2 + delta^2)`; the reference uses the plain `||u||`.

`max_consecutive_restoration = nothing` runs without a restoration limit; the report
checks the at-most-`MAX_RESTORATION_RUN` requirement either way.
"""
Base.@kwdef struct Options
    # -------------------------------------------------------------- problem
    case_id::String = "lyapunov_l1_to_l2"
    execution_mode::Symbol = :cpu                               # one of EXECUTION_MODES
    uniform_mesh_arcs::Union{Nothing,Int} = nothing             # nothing: the reference count
    truncated_uniform_mesh_arcs::Union{Nothing,Int} = nothing   # keep only the first arcs (tests)
    control_norm_eps::Float64 = 1e-6
    mass_flow_smoothing::Float64 = control_norm_eps
    bryson_sigma_factor::Float64 = 3.0
    acceleration_diffusion_km_s32::Float64 = ACCELERATION_DIFFUSION_KM_S32
    gates_proportional_magnitude_std::Float64 = GATES_PROPORTIONAL_MAGNITUDE_STD
    gates_proportional_pointing_std::Float64 = GATES_PROPORTIONAL_POINTING_STD
    navigation_position_std_km::Float64 = 10.0
    navigation_velocity_std_km_s::Float64 = 1.0e-4
    violation_parameter::Float64 = 0.01
    position_covariance_reduction::Float64 = (INITIAL_POSITION_STD_KM / FINAL_POSITION_STD_KM)^2
    velocity_covariance_reduction::Float64 = (INITIAL_VELOCITY_STD_KM_S / FINAL_VELOCITY_STD_KM_S)^2
    cholesky_jitter::Float64 = 1e-12
    eigenvalue_smoothing::Float64 = 1e-6
    spectral_radius_floor::Float64 = 1e-6
    terminal_margin_floor::Float64 = 1e-5
    # -------------------------------------------- stochastic solve (MadNLP)
    tol::Float64 = 1e-6
    acceptable_tol::Float64 = tol
    max_iter::Int = 5000
    alpha_min_frac::Float64 = 0.05
    barrier_mu_init::Float64 = 0.1        # monotone barrier rule
    barrier_mu_min::Float64 = 1e-8
    max_consecutive_restoration::Union{Nothing,Int} = nothing
    condensed_relaxation::Float64 = tol / 10   # :gpu, bound_relax_factor of the relaxed equalities
    # ---------------------------- energy-optimal seed solve (MadNLP, on the CPU)
    nominal_tol::Float64 = 1e-8
    nominal_max_iter::Int = 3000
    nominal_barrier_mu_init::Float64 = 0.1
    # --------------------------------------------------------------- output
    verbose::Bool = true                  # MadNLP iteration log on the terminal
end

"Directory of every file of a run: output/<mode>/<case>."
output_directory(o::Options) = joinpath(OUTPUT_ROOT, String(o.execution_mode), o.case_id)

function validate_options(o::Options)
    check(condition, message) = condition || throw(ArgumentError(message))
    check(o.case_id in CASE_IDS, "unknown case $(o.case_id); expected one of $(CASE_IDS)")
    check(o.execution_mode in EXECUTION_MODES, "execution_mode must be one of $(EXECUTION_MODES)")
    check(o.control_norm_eps > 0 && o.bryson_sigma_factor > 0, "control_norm_eps and bryson_sigma_factor must be positive")
    check(o.mass_flow_smoothing >= 0, "mass_flow_smoothing must be nonnegative")
    check(all(v -> isfinite(v) && v >= 0, (o.acceleration_diffusion_km_s32, o.navigation_position_std_km,
        o.navigation_velocity_std_km_s, o.gates_proportional_magnitude_std, o.gates_proportional_pointing_std)),
        "noise parameters must be finite and nonnegative")
    # The UT oracles implement the reference error model: every error source is present.
    check(o.acceleration_diffusion_km_s32 > 0 &&
          (o.navigation_position_std_km > 0 || o.navigation_velocity_std_km_s > 0) &&
          (o.gates_proportional_magnitude_std > 0 || o.gates_proportional_pointing_std > 0),
        "the UT oracles implement the reference error model (diffusion, navigation and Gates errors)")
    check(0 < o.violation_parameter < 1, "violation_parameter must be in (0, 1)")
    check(0 < o.eigenvalue_smoothing <= 1e-4, "eigenvalue_smoothing must be in (0, 1e-4]")
    check(0 < o.spectral_radius_floor <= 1e-4, "spectral_radius_floor must be in (0, 1e-4]")
    check(0 <= o.terminal_margin_floor <= 1e-4, "terminal_margin_floor must be in [0, 1e-4]")
    check(o.tol > 0 && o.acceptable_tol >= o.tol && o.max_iter > 0, "invalid tolerances or iteration budget")
    check(0 <= o.alpha_min_frac <= 1, "alpha_min_frac must be in [0, 1]")
    check(0 < o.barrier_mu_min < o.barrier_mu_init, "barrier_mu_min must be positive and below barrier_mu_init")
    check(o.max_consecutive_restoration === nothing || 0 <= o.max_consecutive_restoration <= MAX_RESTORATION_RUN,
        "max_consecutive_restoration must be nothing (no limit) or between 0 and $MAX_RESTORATION_RUN")
    check(0 < o.condensed_relaxation <= o.tol, "condensed_relaxation must be positive and at most tol")
    return o
end
