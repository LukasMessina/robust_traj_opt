# Configuration. Problem data are the constants and `Options` defaults of
# stochastic_traj_opt_jax.py; the solver settings are MadNLP's.

# UT dimension: state, navigation error and the Wiener increments of the single RK9 step per arc.
const UT_DIMENSION = NX + NP + NW
# The reference hard-codes p = sqrt(p2/6 + 1e-30) in its 3x3 eigenvalue formula;
# `eigenvalue_smoothing = 1e-15` reproduces that constant bitwise.
const JAX_EIGENVALUE_SMOOTHING = 1e-15
# SNOPT major feasibility tolerance: the reference's acceptance threshold.
const JAX_FEASIBILITY_TOL = 1e-8
# Required bound on consecutive restoration iterations, checked for every run.
const MAX_RESTORATION_RUN = 5

const OUTPUT_ROOT = joinpath(PROJECT_ROOT, "output")
const CR3BP_CASE_IDS = ("lyapunov_l1_to_l2", "nrho_l2_to_dro", "halo_l2_to_halo_l1")
const CASE_IDS = (CR3BP_CASE_IDS..., "earth_mars")

"Exported data of a case (`generated/cases.jl`), including its problem defaults."
function case_entry(case_id::AbstractString)
    haskey(CASE_DATA, case_id) || throw(ArgumentError("unknown case $case_id; expected one of $(CASE_IDS)"))
    return CASE_DATA[case_id]
end
"Problem default of `default_options(case_id)` in the JAX script."
case_default(case_id, name::Symbol) = getfield(case_entry(case_id).defaults, name)
# :cpu  sparse KKT system + MUMPS, exact equalities
# :gpu  condensed KKT system + cuDSS on a CUDA device, equalities relaxed by `condensed_relaxation`
const EXECUTION_MODES = (:cpu, :gpu)
# :spectral  ||S||_eps + psi rho(T) <= 1 with the smoothed largest-eigenvalue formula (the reference)
# :slack     ||S||_eps + psi tau <= 1 and tau^2 I - T = G G' (Cholesky slack, no eigenvalue)
const CHANCE_FORMULATIONS = (:spectral, :slack)

"""
    Options(; kwargs...)

Problem data default to the JAX reference's `default_options(case_id)` (exported per
case), except these robustification settings (reference values in brackets), which
are the same in both execution modes:

* `control_norm_eps = 1e-5`: `eps` in `||S||_eps = sqrt(||S||^2 + eps^2)`, wherever the feedforward
  norm appears (objective, thrust chance constraint, energy-optimal seed solve);
* `gain_bound = 20`: box bound `|K_hat| <= gain_bound` on the scaled feedback gains
  `K_hat = K_tilde / gain_scale` (`Inf` disables it). It is inactive at the optimum of every case
  (largest |K_hat| below 4) but bounds the first interior-point steps, which start from zero
  multipliers on the GPU path and would otherwise move the gains by O(100);
* `control_covariance_weight` (case data: 1 for the CR3BP cases, the original Bryson-rule cost; 1000 for
  Earth-Mars): factor of `tr(T_k)` (executed-control covariance) in the stochastic cost and the TVLQR
  seed's control weight `R~ = control_covariance_weight I`. Unweighted, `tr(T_k)` is ~0.2% of the
  Earth-Mars cost, so its feedback-gain directions are nearly flat and the interior-point solves stall;
* `eigenvalue_smoothing = 1e-6` [1e-15]: `s` in `p = sqrt(p2/6 + s^2)` of the smoothed
  largest-eigenvalue formula of the thrust chance constraint;
* `spectral_radius_floor = 1e-6` [1e-7]: `f` in `sqrt(lambda_max + f^2)`;
* `terminal_margin_floor = 1e-5` [1e-4]: lower bound of the terminal margin factor diagonal;
* `mass_flow_smoothing = control_norm_eps` [0]: `delta` in the sigma-point mass flow
  `T_max sqrt(||e||^2 + delta^2)`; the reference uses the plain `||u||`.

`chance_formulation` selects the thrust chance constraint: `:spectral` (the reference)
is `||S||_eps + psi sqrt(lambda_max(T) + f^2) <= 1` with the smoothed eigenvalue formula;
`:slack` lifts it as in Kumagai & Oguri (2025), `||S||_eps + psi tau <= 1` with
`tau^2 I - T = G G'` (G lower triangular, diagonal >= `chance_slack_floor`), which
needs no eigenvalue.

`max_consecutive_restoration = nothing` runs without a restoration limit; the report
checks the at-most-`MAX_RESTORATION_RUN` requirement either way.
"""
Base.@kwdef struct Options
    # -------------------------------------------------------------- problem
    case_id::String = "lyapunov_l1_to_l2"
    execution_mode::Symbol = :cpu                               # one of EXECUTION_MODES
    uniform_mesh_arcs::Union{Nothing,Int} = nothing             # nothing: the reference count
    truncated_uniform_mesh_arcs::Union{Nothing,Int} = nothing   # keep only the first arcs (tests)
    initial_position_std_km::Float64 = case_default(case_id, :initial_position_std_km)
    initial_velocity_std_km_s::Float64 = case_default(case_id, :initial_velocity_std_km_s)
    initial_mass_std_kg::Float64 = case_default(case_id, :initial_mass_std_kg)
    mass_scale_kg::Float64 = case_default(case_id, :mass_scale_kg)
    control_norm_eps::Float64 = 1e-5
    mass_flow_smoothing::Float64 = control_norm_eps
    bryson_sigma_factor::Float64 = 3.0
    acceleration_diffusion_km_s32::Float64 = case_default(case_id, :acceleration_diffusion_km_s32)
    gates_proportional_magnitude_std::Float64 = case_default(case_id, :gates_proportional_magnitude_std)
    gates_proportional_pointing_std::Float64 = case_default(case_id, :gates_proportional_pointing_std)
    navigation_position_std_km::Float64 = case_default(case_id, :navigation_position_std_km)
    navigation_velocity_std_km_s::Float64 = case_default(case_id, :navigation_velocity_std_km_s)
    violation_parameter::Float64 = case_default(case_id, :violation_parameter)
    position_covariance_reduction::Float64 = case_default(case_id, :position_covariance_reduction)
    velocity_covariance_reduction::Float64 = case_default(case_id, :velocity_covariance_reduction)
    cholesky_jitter::Float64 = 1e-12
    gain_bound::Float64 = 20.0                                  # |K_hat| <= gain_bound; Inf disables
    control_covariance_weight::Float64 = case_default(case_id, :control_covariance_weight)  # tr(T) factor, TVLQR R
    eigenvalue_smoothing::Float64 = 1e-6
    spectral_radius_floor::Float64 = 1e-6
    terminal_margin_floor::Float64 = 1e-5
    chance_formulation::Symbol = :spectral                      # one of CHANCE_FORMULATIONS
    chance_slack_floor::Float64 = 1e-4                          # :slack, floor of the G diagonal (T_max units)
    # -------------------------------------------- stochastic solve (MadNLP)
    tol::Float64 = 1e-6
    acceptable_tol::Float64 = tol
    max_iter::Int = 5000
    alpha_min_frac::Float64 = 0.05
    barrier_mu_init::Float64 = 0.1        # monotone barrier rule
    barrier_mu_min::Float64 = 1e-10
    max_consecutive_restoration::Union{Nothing,Int} = nothing
    condensed_relaxation::Float64 = tol / 100   # :gpu, bound_relax_factor of the relaxed equalities
    # ---------------------------- energy-optimal seed solve (MadNLP, on the CPU)
    nominal_tol::Float64 = 1e-8
    nominal_max_iter::Int = 3000
    nominal_barrier_mu_init::Float64 = 0.1
    # Zero objective: only search for a point satisfying every constraint.
    feasibility_only::Bool = false
    # --------------------------------------------------------------- output
    verbose::Bool = true                  # MadNLP iteration log on the terminal
end

"Directory of every file of a run: output/<mode>/<case>, or <case>_feasibility for a zero-objective run."
output_directory(o::Options) =
    joinpath(OUTPUT_ROOT, String(o.execution_mode), o.case_id * (o.feasibility_only ? "_feasibility" : ""))

function validate_options(o::Options)
    check(condition, message) = condition || throw(ArgumentError(message))
    check(o.case_id in CASE_IDS, "unknown case $(o.case_id); expected one of $(CASE_IDS)")
    check(o.execution_mode in EXECUTION_MODES, "execution_mode must be one of $(EXECUTION_MODES)")
    check(o.control_norm_eps > 0 && o.bryson_sigma_factor > 0, "control_norm_eps and bryson_sigma_factor must be positive")
    check(all(v -> isfinite(v) && v > 0, (o.initial_position_std_km, o.initial_velocity_std_km_s, o.mass_scale_kg)) &&
          isfinite(o.initial_mass_std_kg) && o.initial_mass_std_kg >= 0,
        "initial dispersions and the mass scale must be finite and positive")
    check(o.mass_flow_smoothing >= 0, "mass_flow_smoothing must be nonnegative")
    check(o.gain_bound > 0, "gain_bound must be positive (Inf disables the bound)")
    check(isfinite(o.control_covariance_weight) && o.control_covariance_weight > 0,
        "control_covariance_weight must be positive and finite")
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
    check(o.chance_formulation in CHANCE_FORMULATIONS, "chance_formulation must be one of $(CHANCE_FORMULATIONS)")
    check(0 < o.chance_slack_floor <= 1e-2, "chance_slack_floor must be in (0, 1e-2]")
    check(o.tol > 0 && o.acceptable_tol >= o.tol && o.max_iter > 0, "invalid tolerances or iteration budget")
    check(0 <= o.alpha_min_frac <= 1, "alpha_min_frac must be in [0, 1]")
    check(0 < o.barrier_mu_min < o.barrier_mu_init, "barrier_mu_min must be positive and below barrier_mu_init")
    check(o.max_consecutive_restoration === nothing || 0 <= o.max_consecutive_restoration <= MAX_RESTORATION_RUN,
        "max_consecutive_restoration must be nothing (no limit) or between 0 and $MAX_RESTORATION_RUN")
    check(0 < o.condensed_relaxation <= o.tol, "condensed_relaxation must be positive and at most tol")
    return o
end
