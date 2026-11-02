# Configuration. Problem data are the constants and `Options` defaults of
# stochastic_traj_opt_jax.py; the solver settings are MadNLP's.

# UT dimension: state, navigation error and the Wiener increments of the single RK9 step per arc.
const UT_DIMENSION = NX + NP + NW
# Historical reference value, retained for legacy regression checks. Current
# JAX and MadNLP defaults both use eigenvalue_smoothing = 1e-6.
const JAX_EIGENVALUE_SMOOTHING = 1e-15
# Required bound on consecutive restoration iterations, checked for every run.
const MAX_RESTORATION_RUN = 5

const OUTPUT_ROOT = joinpath(PROJECT_ROOT, "output")
const CR3BP_CASE_IDS = ("lyapunov_l1_to_l2", "nrho_l2_to_dro", "halo_l2_to_halo_l1")
const CASE_IDS = (CR3BP_CASE_IDS..., "earth_mars", "psyche")

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
# :cpu  sparse KKT system + MUMPS, exact equalities
# :gpu  condensed KKT system + cuDSS on a CUDA device, equalities relaxed by `condensed_relaxation`
# :hybrid the model built and evaluated on a CUDA device, MadNLP with the CPU's exact KKT system and MUMPS on the
#        host (`HostModel`: the point goes to the device, the values come back)
const EXECUTION_MODES = (:cpu, :gpu, :hybrid)

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
# :spectral  ||S||_eps + psi rho(T) <= 1 with the smoothed largest-eigenvalue formula (the reference)
# :slack     ||S||_eps + psi tau <= 1 and tau^2 I - T = G G' (Cholesky slack, no eigenvalue)
const CHANCE_FORMULATIONS = (:spectral, :slack)
# :standard  sum of ||S_k||_eps + w_P tr(P_k)/c^2 + w tr(T_k)
# :quantile  sum of the thrust-chance left-hand side, the (1 - beta) quantile bound of the executed
#            thrust magnitude, in the form of `chance_formulation`: :spectral ||S_k||_eps + psi rho(T_k)
#            (smoothed eigenvalue, epigraph variables) or :slack ||S_k||_eps + psi tau_k (Cholesky slack)
# :terminal_dispersion  the level t of the terminal bound G P_N G' <= t I (Psyche): the smallest
#            terminal dispersion relative to the terminal target that the other constraints allow
# :chance_sum  the thrust-chance left-hand sides themselves, sum_k w_k (||S_k|| + psi rho(T_k)) with the arc
#            weights N dt_k / (t_f - t_0), no epigraph variable (`ChanceObjectiveModel`); spectral form only
const OBJECTIVES = (:standard, :quantile, :terminal_dispersion, :chance_sum)
# Terminal covariance: :constraint G P_N G' <= I, or :cost w_N tr(G P_N G')/c^2 added to the standard cost
# (the Bryson weight of the TVLQR seed's terminal Riccati weight; Psyche).
const TERMINAL_COVARIANCE = (:constraint, :cost)
# Covariance defect rows: :product L_{k+1} L_{k+1}' - V_k, or the square-root forms L_{k+1} - R_k (CPU):
# :factor R_k = chol(V_k), :qr R_k from the QR decomposition of the sigma-point deviations (V_k never formed).
const COVARIANCE_DEFECTS = (:product, :factor, :qr)
# Barrier rule of the stochastic solve: monotone Fiacco-McCormick, or the adaptive quality-function and
# LOQO rules of Nocedal, Waechter & Waltz (2009), which fall back to the monotone rule without progress.
const BARRIER_RULES = (:monotone, :quality, :loqo)
# Psyche seed gains: :tvlqr (Bryson TVLQR) or :uniform (every scaled gain entry K_hat = psyche_uniform_gain).
const PSYCHE_SEED_GAINS = (:tvlqr, :uniform)
# Feedforward variables: :cartesian S_k, or :polar (Gamma_k, alpha_k, beta_k) with S_k = Gamma_k d_k,
# Gamma_k >= 0 and d_k = H_k (cos a cos b, sin a cos b, sin b) unit, H_k the reflection taking e1 to
# the seed direction: ||S_k|| = Gamma_k is a variable, so no norm smoothing is needed; or :throttle, the
# deterministic optimisation's throttle Gamma_k in [0, 1] and direction d_k with rows ||d_k||^2 = 1.
const CONTROL_PARAMETERIZATIONS = (:cartesian, :polar, :throttle)

"""
    Options(; kwargs...)

Problem data default to the JAX reference's `default_options(case_id)` (exported per
case). The shared regularisation defaults below match the current JAX script and
are the same in both execution modes:

* `control_norm_eps = 1e-5`: `eps` in `||S||_eps = sqrt(||S||^2 + eps^2)`, wherever the feedforward
  norm appears (objective, thrust chance constraint, energy-optimal seed solve);
* `gain_bound = 50`: box bound `|K_hat| <= gain_bound` on the scaled feedback gains
  `K_hat = K_tilde / gain_scale` (`Inf` disables it). It is inactive at the optimum of every case
  (largest |K_hat| below 4) but bounds the first interior-point steps, which start from zero
  multipliers on the GPU path and would otherwise move the gains by O(100);
* `control_covariance_weight` (`TRACKING_WEIGHTS`, else the case data: 1 for CR3BP, 1000 for
  Earth-Mars and Psyche): factor of `tr(T_k)` (executed-control covariance) in the stochastic cost and the TVLQR
  seed's control weight `R~ = control_covariance_weight I`. Unweighted, `tr(T_k)` is ~0.1-0.3% of the
  Earth-Mars and Psyche costs, so its feedback-gain directions are nearly flat and the interior-point solves stall;
* `state_covariance_weight` (`TRACKING_WEIGHTS`, else 1): factor `w_P` of `tr(P_k[1:6,1:6])/c^2` in the
  stochastic cost and of the TVLQR seed's running state weight `Q~ = w_P I/c^2` (the terminal seed weight
  stays `diag(r_p I3, r_v I3)/c^2`);
* `eigenvalue_smoothing = 1e-6`: `s` in `p = sqrt(p2/6 + s^2)` of the smoothed
  largest-eigenvalue formula of the thrust chance constraint;
* `spectral_radius_floor = 1e-6`: `f` in `sqrt(lambda_max + f^2)`;
* `terminal_margin_floor = 1e-5`: lower bound of the terminal margin factor diagonal;
* `mass_flow_smoothing = control_norm_eps`: `delta` in the sigma-point mass flow
  `T_max sqrt(||e||^2 + delta^2)`; zero selects the plain norm;
* `model_mass_flow_smoothing = 0`: the same `delta` in the sigma-point mass flow of the polar and
  throttle models, which otherwise use the plain norm, on the outer points whose command carries feedback only
  (the others command Gamma d and keep the exact norm; `outer_flow_sq`); ignored by the cartesian model;
* `sigma_command_lifting = false`: throttle controls only; the commands of the outer sigma points that
  carry feedback become variables c_o = eta_o e_o (||e_o|| = 1, eta_o free) tied to Gamma d + sigma K m_o
  by equality rows, so the mass flow and the Gates factors see eta_o and the unit e_o (exact, no delta);
* `lifting_start = :pushed`: with lifted commands, Gamma and the factors start at MadNLP's pushed initial point
  (`bound_push`) and (eta, e) from it; `:seed` keeps the seed values (MadNLP then moves the unlifted points only);
* `bound_push = 1e-2`: MadNLP's `bound_push` of the stochastic solve (its default).

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
    # Single-phase cases: the deterministic reference (:energy or :fuel, the fuel one with the throttle controls,
    # also the objective of the reoptimisation) and, in place of the uniform mesh, the reference's own mesh with
    # every interval split into this many equal pieces (arc cost weights N dt_k / (t_f - t_0) then).
    reference_objective::Symbol = case_id == "psyche" ? :energy : :fuel
    # Single-phase TVLQR seed weights (nothing: the cost weights), e.g. Bryson's rule w_P = w = 1 with other cost weights
    seed_state_covariance_weight::Union{Nothing,Float64} = nothing
    seed_control_covariance_weight::Union{Nothing,Float64} = nothing
    reference_mesh_refinement::Union{Nothing,Int} =
        case_id == "psyche" || uniform_mesh_arcs !== nothing || truncated_uniform_mesh_arcs !== nothing ? nothing : 2
    # Psyche uses the initial position and mass standard deviations and the navigation errors
    # below; its velocity covariance comes from the export, its terminal target from
    # psyche_terminal_*_std (not the reductions).
    initial_position_std_km::Float64 = case_default(case_id, :initial_position_std_km)
    initial_velocity_std_km_s::Float64 = case_default(case_id, :initial_velocity_std_km_s)
    initial_mass_std_kg::Float64 = case_default(case_id, :initial_mass_std_kg)
    mass_scale_kg::Float64 = case_default(case_id, :mass_scale_kg)
    control_norm_eps::Float64 = 1e-5
    mass_flow_smoothing::Float64 = control_norm_eps
    model_mass_flow_smoothing::Float64 = 5e-4
    sigma_command_lifting::Bool = false
    lifting_start::Symbol = :pushed
    bound_push::Float64 = 1e-2
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
    gain_bound::Float64 = Inf                                   # |K_hat| <= gain_bound; Inf disables
    control_covariance_weight::Float64 = case_default(case_id, :control_covariance_weight)  # tr(T) factor, TVLQR R
    state_covariance_weight::Float64 = case_default(case_id, :state_covariance_weight)      # tr(P)/c^2 factor, TVLQR Q
    eigenvalue_smoothing::Float64 = 1e-6
    spectral_radius_floor::Float64 = 1e-6
    terminal_margin_floor::Float64 = 1e-5
    chance_formulation::Symbol = :spectral                      # one of CHANCE_FORMULATIONS
    objective::Symbol = :standard                               # one of OBJECTIVES
    terminal_covariance::Symbol = :cost                         # one of TERMINAL_COVARIANCE
    terminal_covariance_weight::Float64 = 1.0                   # :cost, w_N of w_N tr(G P_N G')/c^2
    covariance_defect::Symbol = :factor                         # one of COVARIANCE_DEFECTS
    norm_epigraph::Bool = false                                 # s_k >= ||S_k|| in place of ||S_k||_eps
    control_parameterization::Symbol = :throttle                # one of CONTROL_PARAMETERIZATIONS
    # :throttle direction rows: :sphere ||d||^2 = 1 (the deterministic optimisation), or :ball ||d||^2 <= 1
    # (lossless: with the cost on Gamma a thrust S is cheapest with ||d|| = 1; coast directions stay interior)
    direction_constraint::Symbol = :sphere
    chance_slack_floor::Float64 = 1e-4                          # :slack, floor of the G diagonal (T_max units)
    # -------------------------------------------- stochastic solve (MadNLP)
    tol::Float64 = 1e-6
    acceptable_tol::Float64 = tol
    max_iter::Int = 5000
    alpha_min_frac::Float64 = 0.05
    barrier_mu_init::Float64 = 0.1        # monotone barrier rule
    barrier_mu_min::Float64 = 1e-10
    barrier_rule::Symbol = :monotone      # stochastic solve: one of BARRIER_RULES
    # Least-squares initial multipliers are kept up to this infinity norm and otherwise all set to zero
    # (MadNLP's default 1e3). With zero multipliers the first step sees only the objective's curvature.
    constr_mult_init_max::Float64 = 1e4
    max_consecutive_restoration::Union{Nothing,Int} = nothing
    condensed_relaxation::Float64 = tol / 100   # :gpu, bound_relax_factor of the relaxed equalities
    # ---------------------------- energy-optimal seed solve (MadNLP, on the CPU)
    nominal_tol::Float64 = 1e-8
    nominal_max_iter::Int = 3000
    nominal_barrier_mu_init::Float64 = 0.1
    # ------------------------------------------------------------- Psyche
    # Mission arrays exported from the deterministic energy solution (one RK9 step per interval).
    psyche_input_file::String = joinpath(PROJECT_ROOT, "data", "psyche_inputs_fuel_refine2.npz")
    # Objective of the reoptimisation on the stochastic mesh: :energy (0.5 sum dt ||T||^2) or :fuel (the deterministic
    # optimisation's 1 - m_f / m_wet; throttle controls only), matching the deterministic reference it starts from.
    psyche_nominal_objective::Symbol = control_parameterization == :throttle ? :fuel : :energy
    # Terminal covariance bound P_N <= diag(sp^2 I3, sv^2 I3) at the asteroid rendezvous.
    psyche_terminal_position_std_km::Float64 = 1500.0
    psyche_terminal_velocity_std_km_s::Float64 = 0.002
    # Seed gains (one of PSYCHE_SEED_GAINS); :uniform sets every scaled entry K_hat of the powered arcs.
    psyche_seed_gain::Symbol = :tvlqr
    # TVLQR seed weights (nothing: the cost weights), so the cost can be retuned with the same seed.
    psyche_seed_state_covariance_weight::Union{Nothing,Float64} = nothing
    psyche_seed_control_covariance_weight::Union{Nothing,Float64} = nothing
    psyche_uniform_gain::Float64 = -0.01
    # Its already-nearly-feasible energy seed needs a smaller initial barrier.
    psyche_nominal_barrier_mu_init::Float64 = 1e-6
    # Zero objective: only search for a point satisfying every constraint.
    feasibility_only::Bool = false
    # --------------------------------------------------------------- output
    verbose::Bool = true                  # MadNLP iteration log on the terminal
    # Monte Carlo validation after the solve (tools/validate*.py, JAX on the solve's device)
    validate_after_solve::Bool = true
    validation_python::String = default_validation_python()
end

"""
Directory of every file of a run: output/<mode>/<case>, <case>_feasibility for a zero-objective run,
<case>_terminal_dispersion for the minimum terminal dispersion or <case>_terminal_cost for a
terminal covariance cost in place of the bound.
"""
output_directory(o::Options) =
    joinpath(get(ENV,"MADNLP_STOCHASTIC_OUTPUT_ROOT",OUTPUT_ROOT), String(o.execution_mode), o.case_id *
        (o.feasibility_only ? "_feasibility" : o.objective == :terminal_dispersion ? "_terminal_dispersion" :
         o.terminal_covariance == :cost ? "_terminal_cost" : o.objective == :chance_sum ? "_chance_sum" : "") *
        (o.covariance_defect == :product ? "" : "_" * String(o.covariance_defect)) *
        (o.control_parameterization == :cartesian ? "" : "_" * String(o.control_parameterization)))

function validate_options(o::Options)
    check(condition, message) = condition || throw(ArgumentError(message))
    check(o.case_id in CASE_IDS, "unknown case $(o.case_id); expected one of $(CASE_IDS)")
    check(o.psyche_nominal_barrier_mu_init > 0, "the Psyche nominal barrier must be positive")
    check(o.psyche_terminal_position_std_km > 0 && o.psyche_terminal_velocity_std_km_s > 0,
        "the Psyche terminal standard deviations must be positive")
    check(o.case_id != "psyche" || o.chance_formulation == :spectral || o.control_parameterization != :cartesian,
        "Psyche supports the slack chance constraint with polar or throttle controls")
    check(o.execution_mode in EXECUTION_MODES, "execution_mode must be one of $(EXECUTION_MODES)")
    check(o.objective in OBJECTIVES, "objective must be one of $(OBJECTIVES)")
    check(o.objective != :terminal_dispersion || (o.case_id == "psyche" && !o.feasibility_only),
        "the terminal_dispersion objective is a Psyche solve with an objective (not --feasibility)")
    check(o.terminal_covariance in TERMINAL_COVARIANCE, "terminal_covariance must be one of $(TERMINAL_COVARIANCE)")
    check(o.terminal_covariance == :constraint || o.objective == :standard,
        "terminal_covariance = :cost is an option of the standard objective")
    check(o.reference_objective in (:energy, :fuel), "reference_objective must be :energy or :fuel")
    check(o.reference_objective == :energy || (o.case_id != "psyche" && o.control_parameterization == :throttle),
        "the fuel reference of a single-phase case uses the throttle controls (Psyche: --psyche-input-file)")
    check(o.reference_mesh_refinement === nothing || (o.reference_mesh_refinement >= 1 && o.case_id != "psyche" &&
          o.uniform_mesh_arcs === nothing && o.truncated_uniform_mesh_arcs === nothing),
        "reference_mesh_refinement is a positive integer for a single-phase case, without the uniform-mesh options")
    check(isfinite(o.terminal_covariance_weight) && o.terminal_covariance_weight > 0,
        "terminal_covariance_weight must be positive and finite")
    check(o.psyche_seed_gain in PSYCHE_SEED_GAINS, "psyche_seed_gain must be one of $(PSYCHE_SEED_GAINS)")
    check(o.covariance_defect in COVARIANCE_DEFECTS, "covariance_defect must be one of $(COVARIANCE_DEFECTS)")
    check(o.barrier_rule in BARRIER_RULES, "barrier_rule must be one of $(BARRIER_RULES)")
    check(!o.norm_epigraph || (o.objective == :standard && o.chance_formulation == :spectral),
        "norm_epigraph needs the standard objective and the spectral chance constraint")
    check(o.covariance_defect != :qr || o.execution_mode == :cpu, "covariance_defect = :qr runs on the CPU")
    check(o.control_parameterization in CONTROL_PARAMETERIZATIONS,
        "control_parameterization must be one of $(CONTROL_PARAMETERIZATIONS)")
    check(o.objective != :chance_sum || (!o.norm_epigraph && (o.chance_formulation == :spectral || o.control_parameterization != :cartesian)),
        "the chance_sum objective uses the spectral chance rows, or the slack rows with polar or throttle controls, " *
        "without the norm epigraph")
    check(o.control_parameterization == :cartesian || (o.objective in (:standard, :chance_sum) &&
          !o.norm_epigraph && !o.feasibility_only), "the polar and throttle controls need the standard or chance_sum objective " *
        "and no norm epigraph")
    check(o.direction_constraint in (:sphere, :ball), "direction_constraint must be :sphere or :ball")
    check(o.psyche_nominal_objective in (:energy, :fuel), "psyche_nominal_objective must be :energy or :fuel")
    check(o.psyche_nominal_objective == :energy || o.control_parameterization == :throttle,
        "the fuel reoptimisation uses the throttle controls")
    check(abs(o.psyche_uniform_gain) < o.gain_bound, "psyche_uniform_gain must lie inside the gain bound")
    check(o.control_norm_eps > 0 && o.bryson_sigma_factor > 0, "control_norm_eps and bryson_sigma_factor must be positive")
    check(all(v -> isfinite(v) && v > 0, (o.initial_position_std_km, o.initial_velocity_std_km_s, o.mass_scale_kg)) &&
          isfinite(o.initial_mass_std_kg) && o.initial_mass_std_kg >= 0,
        "initial dispersions and the mass scale must be finite and positive")
    check(o.mass_flow_smoothing >= 0, "mass_flow_smoothing must be nonnegative")
    # the cartesian model ignores it (it uses mass_flow_smoothing)
    check(isfinite(o.model_mass_flow_smoothing) && o.model_mass_flow_smoothing >= 0,
        "model_mass_flow_smoothing must be finite and nonnegative")
    check(!o.sigma_command_lifting || o.control_parameterization == :throttle,
        "sigma_command_lifting applies to the throttle controls")
    check(o.lifting_start in (:pushed, :seed), "lifting_start must be :pushed or :seed")
    check(isfinite(o.bound_push) && o.bound_push > 0, "bound_push must be positive and finite")
    check(o.gain_bound > 0, "gain_bound must be positive (Inf disables the bound)")
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
