# Configuration. Problem data are the constants and `Options` defaults of
# stochastic_traj_opt_jax.py; the solver settings are MadNLP's.
#
# There are two problems, selected by `objective`:
#   :standard   (step 3) the tracking cost sum ||S_k|| + w_P tr(P_k)/c^2 + w tr(T_k) with the terminal covariance
#               in the cost, the thrust chance rows Gamma_k + psi rho(T_k) <= 1 (smoothed largest eigenvalue) and
#               the TVLQR seed weighted like the cost;
#   :chance_sum (step 4) the time-weighted sum of the chance left-hand sides through the slack epigraph
#               tau_hat^2 I - G_hat G_hat' - T/s^2 = 0, the hard terminal covariance bound and Bryson's seed weights
#               (Psyche: the cost's).
# Both use the throttle Gamma_k in [0, 1] and unit direction d_k of the deterministic optimisation, square-root
# covariance defects and the fuel-optimal reference.

# UT dimension: state, navigation error and the Wiener increments of the single RK9 step per arc.
const UT_DIMENSION = NX + NP + NW
# Required bound on consecutive restoration iterations, checked for every run.
const MAX_RESTORATION_RUN = 5

# Fixed formulation constants.
const BRYSON_SIGMA_FACTOR = 3.0          # c of the Bryson weights tr(P)/c^2
const CHOLESKY_JITTER = 1e-12            # added to the covariance before its Cholesky factor (seed, Gates)
const CONTROL_NORM_EPS = 1e-5            # eps of the JAX reference's ||S||_eps; written for its Monte Carlo replay
const MASS_FLOW_SMOOTHING = 5e-4         # delta of the sigma-point mass flow on the feedback-carrying outer points
const EIGENVALUE_SMOOTHING = 1e-6        # s in p = sqrt(p2/6 + s^2) of the largest-eigenvalue formula
const SPECTRAL_RADIUS_FLOOR = 1e-6       # f in sqrt(lambda_max + f^2)
const TERMINAL_MARGIN_FLOOR = 1e-5       # floor of the terminal margin factor diagonal (step 4)
const TERMINAL_COVARIANCE_WEIGHT = 1.0   # w_N of the terminal covariance cost w_N tr(G P_N G')/c^2 (step 3)
const CHANCE_SLACK_FLOOR = 1e-4          # floor of the slack factor diagonal, T_max units (step 4)
const REFERENCE_MESH_REFINEMENT = 2      # single-phase mesh: every reference interval split into 2

const OUTPUT_ROOT = joinpath(PROJECT_ROOT, "output")
const CR3BP_CASE_IDS = ("lyapunov_l1_to_l2", "nrho_l2_to_dro", "halo_l2_to_halo_l1")
const CASE_IDS = (CR3BP_CASE_IDS..., "earth_mars", "psyche")
# Psyche mission arrays exported from the deterministic fuel-optimal solution (reference mesh refined x2).
const PSYCHE_INPUT_FILE = joinpath(PROJECT_ROOT, "data", "psyche_inputs_fuel_refine2.npz")

"Exported data of a case (`generated/cases.jl`), including its problem defaults."
function case_entry(case_id::AbstractString)
    haskey(CASE_DATA, case_id) || throw(ArgumentError("unknown case $case_id; expected one of $(CASE_IDS)"))
    return CASE_DATA[case_id]
end
# Tracking weights of the standard cost and the TVLQR seed, per case: state_covariance_weight w_P
# (factor of tr(P)/c^2, seed Q~ = w_P I/c^2) and control_covariance_weight w (factor of tr(T),
# seed R~ = w I). Tuned for a 92 / 2 / 6 % split of the objective (mean control / state
# covariance / control covariance); the same values as default_options in the JAX script.
const TRACKING_WEIGHTS = Dict{String,NamedTuple}(
    "lyapunov_l1_to_l2" => (state_covariance_weight = 0.019944701695746685, control_covariance_weight = 33.55903293181279),
    "halo_l2_to_halo_l1" => (state_covariance_weight = 0.2128404007964649, control_covariance_weight = 118.38122955698798),
    "nrho_l2_to_dro" => (state_covariance_weight = 0.15314443530196445, control_covariance_weight = 129.24846631819943),
    "earth_mars" => (state_covariance_weight = 0.1530371475171174, control_covariance_weight = 151.8285791060962))

# Psyche problem data that differ from Earth-Mars: the initial position standard deviation
# per axis at Earth SOI exit (its velocity covariance is the launch covariance of the export), and the
# tracking weights of a mean-control share of 95 % (94.8 / 1.2 / 4.0 % with the fuel start and delta 5e-4,
# seed = cost weights; CPU 432 iterations, no restoration).
const PSYCHE_DEFAULTS = (; initial_position_std_km = 2500.0, state_covariance_weight = 0.0102, control_covariance_weight = 54.3)

"Problem default of `default_options(case_id)` in the JAX script."
# Psyche takes every other problem default from Earth-Mars.
function case_default(case_id, name::Symbol)
    weights = get(TRACKING_WEIGHTS, case_id, (;))
    haskey(weights, name) && return getfield(weights, name)
    case_id == "psyche" && haskey(PSYCHE_DEFAULTS, name) && return getfield(PSYCHE_DEFAULTS, name)
    name == :state_covariance_weight && return 1.0
    return getfield(case_entry(case_id=="psyche" ? "earth_mars" : case_id).defaults, name)
end
# :cpu  the model and MadNLP (sparse KKT system + MUMPS, exact equalities) on the CPU
# :gpu  the model built and evaluated on a CUDA device, MadNLP with the same exact KKT system and MUMPS on the
#       host (`HostModel`: the point goes to the device, the values come back)
const EXECUTION_MODES = (:cpu, :gpu)
# :standard (step 3) or :chance_sum (step 4); see the top of this file.
const OBJECTIVES = (:standard, :chance_sum)

"""
Python of the post-solve validation: MADNLP_VALIDATION_PYTHON, else PSYCHE_PYTHON, else
the conda environment `project` (JAX, CasADi, SciPy, Matplotlib; also used by
stochastic_optimization/wsl_snopt_python.sh) when installed, else `python3`.
"""
function default_validation_python()
    for name in ("MADNLP_VALIDATION_PYTHON", "PSYCHE_PYTHON")
        haskey(ENV, name) && return ENV[name]
    end
    project = joinpath(homedir(), "anaconda3", "envs", "project", "bin", "python")
    return isfile(project) ? project : "python3"
end

"""
    Options(; kwargs...)

The case, the execution mode and the problem (`objective`). The problem data default to the JAX
reference's `default_options(case_id)` (exported per case); the tracking weights `state_covariance_weight`
`w_P` (factor of `tr(P_k[1:6,1:6])/c^2`) and `control_covariance_weight` `w` (factor of `tr(T_k)`) are the
per-case `TRACKING_WEIGHTS` and `PSYCHE_DEFAULTS`. The formulation constants are at the top of this file.
"""
Base.@kwdef struct Options
    # -------------------------------------------------------------- problem
    case_id::String = "lyapunov_l1_to_l2"
    execution_mode::Symbol = :cpu                               # one of EXECUTION_MODES
    objective::Symbol = :standard                               # one of OBJECTIVES
    truncated_arcs::Union{Nothing,Int} = nothing                # single-phase: keep only the first arcs (tests)
    # Psyche uses the initial position and mass standard deviations and the navigation errors
    # below; its velocity covariance comes from the export, its terminal target from
    # psyche_terminal_*_std (not the reductions).
    initial_position_std_km::Float64 = case_default(case_id, :initial_position_std_km)
    initial_velocity_std_km_s::Float64 = case_default(case_id, :initial_velocity_std_km_s)
    initial_mass_std_kg::Float64 = case_default(case_id, :initial_mass_std_kg)
    mass_scale_kg::Float64 = case_default(case_id, :mass_scale_kg)
    acceleration_diffusion_km_s32::Float64 = case_default(case_id, :acceleration_diffusion_km_s32)
    gates_proportional_magnitude_std::Float64 = case_default(case_id, :gates_proportional_magnitude_std)
    gates_proportional_pointing_std::Float64 = case_default(case_id, :gates_proportional_pointing_std)
    navigation_position_std_km::Float64 = case_default(case_id, :navigation_position_std_km)
    navigation_velocity_std_km_s::Float64 = case_default(case_id, :navigation_velocity_std_km_s)
    violation_parameter::Float64 = case_default(case_id, :violation_parameter)
    position_covariance_reduction::Float64 = case_default(case_id, :position_covariance_reduction)
    velocity_covariance_reduction::Float64 = case_default(case_id, :velocity_covariance_reduction)
    control_covariance_weight::Float64 = case_default(case_id, :control_covariance_weight)  # tr(T) factor, TVLQR R
    state_covariance_weight::Float64 = case_default(case_id, :state_covariance_weight)      # tr(P)/c^2 factor, TVLQR Q
    # Psyche terminal covariance target P_N <= diag(sp^2 I3, sv^2 I3) at the asteroid rendezvous.
    psyche_terminal_position_std_km::Float64 = 1500.0
    psyche_terminal_velocity_std_km_s::Float64 = 0.002
    # -------------------------------------------- stochastic solve (MadNLP)
    tol::Float64 = 1e-6
    acceptable_tol::Float64 = tol
    max_iter::Int = objective == :chance_sum ? 10000 : 5000
    # --------------------------------------------------------------- output
    verbose::Bool = true                  # MadNLP iteration log on the terminal
    # Monte Carlo validation after the solve (tools/validate*.py, JAX on the solve's device)
    validate_after_solve::Bool = true
    validation_python::String = default_validation_python()
end

"Step 4: the chance-sum cost on the slack epigraph, the terminal covariance bound and its seed weights."
chance_sum(o::Options) = o.objective == :chance_sum

"""
TVLQR seed weights (w_P, w): the cost's (step 3, and Psyche's step 4), or Bryson's rule w_P = w = 1 (the
single-phase cases' step 4; from Bryson's seed the Psyche step 4 solve stalls at the iteration limit).
"""
seed_weights(o::Options) = chance_sum(o) && o.case_id != "psyche" ? (1.0, 1.0) :
    (o.state_covariance_weight, o.control_covariance_weight)

"Directory of every file of a run: output/<mode>/<case> (step 3) or output/<mode>/<case>_chance_sum (step 4)."
output_directory(o::Options) = joinpath(get(ENV, "MADNLP_STOCHASTIC_OUTPUT_ROOT", OUTPUT_ROOT), String(o.execution_mode),
    o.case_id * (chance_sum(o) ? "_chance_sum" : ""))

function validate_options(o::Options)
    check(condition, message) = condition || throw(ArgumentError(message))
    check(o.case_id in CASE_IDS, "unknown case $(o.case_id); expected one of $(CASE_IDS)")
    check(o.execution_mode in EXECUTION_MODES, "execution_mode must be one of $(EXECUTION_MODES)")
    check(o.objective in OBJECTIVES, "objective must be one of $(OBJECTIVES)")
    check(o.truncated_arcs === nothing || (o.truncated_arcs >= 1 && o.case_id != "psyche"),
        "truncated_arcs is a positive number of arcs of a single-phase case")
    check(o.psyche_terminal_position_std_km > 0 && o.psyche_terminal_velocity_std_km_s > 0,
        "the Psyche terminal standard deviations must be positive")
    check(all(v -> isfinite(v) && v > 0, (o.initial_position_std_km, o.initial_velocity_std_km_s, o.mass_scale_kg)) &&
          isfinite(o.initial_mass_std_kg) && o.initial_mass_std_kg >= 0,
        "initial dispersions and the mass scale must be finite and positive")
    check(isfinite(o.control_covariance_weight) && o.control_covariance_weight > 0,
        "control_covariance_weight must be positive and finite")
    check(isfinite(o.state_covariance_weight) && o.state_covariance_weight > 0,
        "state_covariance_weight must be positive and finite")
    check(all(v -> isfinite(v) && v >= 0, (o.acceleration_diffusion_km_s32, o.navigation_position_std_km,
        o.navigation_velocity_std_km_s, o.gates_proportional_magnitude_std, o.gates_proportional_pointing_std)),
        "noise parameters must be finite and nonnegative")
    # The UT oracles implement the reference error model: every error source is present.
    check(o.acceleration_diffusion_km_s32 > 0 &&
          (o.navigation_position_std_km > 0 || o.navigation_velocity_std_km_s > 0) &&
          (o.gates_proportional_magnitude_std > 0 || o.gates_proportional_pointing_std > 0),
        "the UT oracles implement the reference error model (diffusion, navigation and Gates errors)")
    check(0 < o.violation_parameter < 1, "violation_parameter must be in (0, 1)")
    check(o.tol > 0 && o.acceptable_tol >= o.tol && o.max_iter > 0, "invalid tolerances or iteration budget")
    return o
end
