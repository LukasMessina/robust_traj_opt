const CLI_HELP = """
Usage: julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl [options]

  --mode=cpu|gpu                    cpu: sparse KKT system + MUMPS (default)
                                    gpu: condensed KKT system + cuDSS on a CUDA device
  --case=lyapunov_l1_to_l2|nrho_l2_to_dro|halo_l2_to_halo_l1|earth_mars|psyche|cr3bp|all
                                    (default lyapunov_l1_to_l2; cr3bp: the three CR3BP cases)
  --quiet                           no MadNLP iteration log on the terminal
  --no-validation                   skip the Monte Carlo validation after the solve
  --validation-python=PATH          Python with JAX for the validation
                                    (default: \$MADNLP_VALIDATION_PYTHON, else \$PSYCHE_PYTHON,
                                    else ~/anaconda3/envs/project/bin/python, else python3)
  --feasibility                     zero objective: only find a feasible point
                                    (results in output/<mode>/<case>_feasibility/)
  --help

Psyche (multiphase Sun-Mars-Sun, one RK9 step per interval):
  --psyche-input-file=PATH          offline exported mission inputs (data/psyche_inputs.npz)
  --reference-objective=energy|fuel  single-phase cases: deterministic reference and reoptimisation objective
                                    [energy]; fuel needs --control-parameterization=throttle
  --seed-state-covariance-weight=W, --seed-control-covariance-weight=W  single-phase TVLQR seed weights
                                    [the cost weights]; Bryson's rule: 1 and 1
  --reference-mesh-refinement=R     single-phase cases: the reference's own mesh, every interval split into R
                                    (in place of the uniform mesh; cost weights N dt_k / (t_f - t_0))
  --psyche-nominal-objective=energy|fuel  reoptimisation on the stochastic mesh [energy]; fuel = 1 - m_f/m_wet
                                    (throttle controls), for inputs exported from the fuel-optimal reference
  --initial-position-std-km=2500    initial position standard deviation per axis at Earth SOI exit
  --psyche-terminal-position-std-km=1500  terminal covariance bound, position per axis
  --psyche-terminal-velocity-std-km-s=0.002  terminal covariance bound, velocity per axis
  --state-covariance-weight=W       factor of tr(P)/c^2 in the cost and TVLQR seed Q
  --psyche-seed-state-covariance-weight=W, --psyche-seed-control-covariance-weight=W
                                    TVLQR seed weights (default: the cost weights)
  --psyche-seed-gain=tvlqr|uniform  seed gains: Bryson TVLQR [tvlqr], or every scaled entry
                                    K_hat of the powered arcs = --psyche-uniform-gain [-0.01]
  --terminal-covariance=constraint|cost  terminal bound G P_N G' <= I [constraint], or the cost
                                    term w_N tr(G P_N G')/c^2 (--terminal-covariance-weight=w_N [1];
                                    results in output/<mode>/psyche_terminal_cost/)
  --norm-epigraph                   thrust norm as epigraph variables s_k >= ||S_k|| (objective and
                                    chance rows) in place of the smoothed ||S_k||_eps
  --sigma-command-lifting           throttle controls: the feedback commands of the outer sigma points as
                                    variables eta e (||e|| = 1) tied to Gamma d + sigma K m by rows
  --lifting-start=pushed|seed       lifted start: MadNLP's pushed point [pushed] or the seed values
  --bound-push=1e-2                 MadNLP bound_push of the stochastic solve [1e-2, MadNLP's default]
  --covariance-defect=product|factor|qr  covariance rows L L' - V [product], or the square-root
                                    form L - R: factor R = chol(V) (CPU and GPU), qr R from the QR of the
                                    sigma-point deviations (CPU; results in output/<mode>/<case>..._<form>/)
  --control-parameterization=cartesian|polar|throttle  feedforward S_k [cartesian]; polar S_k = Gamma_k d_k
                                    with Gamma_k >= 0 and a unit direction of two angles; throttle Gamma_k in
                                    [0, 1] and direction d_k with ||d_k||^2 = 1 rows. ||S_k|| = Gamma_k: no
                                    eps in the objective and chance rows (results in ..._polar/, ..._throttle/)
  --direction-constraint=sphere|ball  throttle direction rows ||d_k||^2 = 1 [sphere] or ||d_k||^2 <= 1 (ball:
                                    lossless, coast directions stay interior)
  --psyche-nominal-barrier-mu-init=1e-6  energy-reoptimisation barrier only
  --nominal-max-iter=3000          iteration limit of the energy reoptimisation

Every result, log and diagnostic goes to output/<mode>/<case>/. After each solve,
tools/validate.py (validate_psyche.py for Psyche) runs the Monte Carlo validation
and figures with JAX on the same device: --mode=cpu on the CPU, --mode=gpu on CUDA.

Problem regularisation (current JAX defaults in brackets):
  --control-norm-eps=1e-5           eps in ||S||_eps = sqrt(||S||^2 + eps^2), seed and stochastic solves [1e-5]
  --gain-bound=20                   |K_hat| <= B on the scaled feedback gains, Inf disables [20]
  --control-covariance-weight=100   factor of tr(control covariance) in the cost and TVLQR seed R [CR3BP 1, Earth-Mars/Psyche 1000]
  --mass-flow-smoothing=D           delta in the sigma-point mass flow T_max sqrt(||e||^2 + delta^2)
                                    (default: --control-norm-eps) [same; zero selects plain norm]
  --eigenvalue-smoothing=1e-6       s in p = sqrt(p2/6 + s^2), at most 1e-4 [1e-6]
  --spectral-radius-floor=1e-6      f in sqrt(lambda_max + f^2), at most 1e-4 [1e-6]
  --terminal-margin-floor=1e-5      terminal margin factor diagonal floor, 0 to 1e-4 [1e-5]
  --chance-formulation=spectral|slack
                                    thrust chance constraint: smoothed largest eigenvalue [spectral],
                                    or tau with tau^2 I - T = G G' (Kumagai & Oguri 2025)
  --chance-slack-floor=1e-4         slack: floor of the G diagonal (T_max units), at most 1e-2
  --objective=standard|quantile|terminal_dispersion
                                    standard: sum ||S||_eps + tr(P)/c^2 + w tr(T) [default];
                                    quantile: sum of the thrust-chance quantile, as the chance rows:
                                    spectral ||S||_eps + psi rho(T), slack ||S||_eps + psi tau
                                    (Psyche duration-weighted; Psyche: spectral only);
                                    terminal_dispersion (Psyche): minimize t in G P_N G' <= t I, G the
                                    terminal target (results in output/<mode>/psyche_terminal_dispersion/)

MadNLP, stochastic solve:
  --tol=1e-6  --acceptable-tol=TOL  --max-iter=5000  --alpha-min-frac=0.05
  --barrier-mu-init=0.1  --barrier-mu-min=1e-10   (monotone barrier rule)
  --barrier-rule=monotone|quality|loqo  stochastic solve: monotone [default], or the adaptive
                                    quality-function / LOQO rules (monotone fallback)
  --constr-mult-init-max=1e3        stochastic solve: least-squares initial multipliers are kept up to
                                    this norm, otherwise all start at zero [MadNLP's 1e3]
  --max-restoration=none|0..5       consecutive restoration limit (default none; the report
                                    always checks the <= 5 requirement)
  --condensed-relaxation=R          gpu: bound_relax_factor of the relaxed equalities
                                    (default TOL/100 = 1e-8 at the default TOL, at most TOL)
"""

real_option(field) = (field, v -> parse(Float64, v))

const CLI_OPTIONS = Dict(
    "--psyche-input-file" => (:psyche_input_file,String),
    "--psyche-nominal-objective" => (:psyche_nominal_objective, Symbol),
    "--reference-objective" => (:reference_objective, Symbol),
    "--seed-state-covariance-weight" => real_option(:seed_state_covariance_weight),
    "--seed-control-covariance-weight" => real_option(:seed_control_covariance_weight),
    "--reference-mesh-refinement" => (:reference_mesh_refinement, v -> parse(Int, v)),
    "--initial-position-std-km" => real_option(:initial_position_std_km),
    "--psyche-terminal-position-std-km" => real_option(:psyche_terminal_position_std_km),
    "--psyche-terminal-velocity-std-km-s" => real_option(:psyche_terminal_velocity_std_km_s),
    "--nominal-max-iter" => (:nominal_max_iter,v->parse(Int,v)),
    "--psyche-seed-gain" => (:psyche_seed_gain, Symbol),
    "--psyche-seed-state-covariance-weight" => real_option(:psyche_seed_state_covariance_weight),
    "--psyche-seed-control-covariance-weight" => real_option(:psyche_seed_control_covariance_weight),
    "--psyche-uniform-gain" => real_option(:psyche_uniform_gain),
    "--terminal-covariance" => (:terminal_covariance, Symbol),
    "--covariance-defect" => (:covariance_defect, Symbol),
    "--control-parameterization" => (:control_parameterization, Symbol),
    "--direction-constraint" => (:direction_constraint, Symbol),
    "--terminal-covariance-weight" => real_option(:terminal_covariance_weight),
    "--psyche-nominal-barrier-mu-init" => real_option(:psyche_nominal_barrier_mu_init),
    "--mode" => (:execution_mode, Symbol),
    "--validation-python" => (:validation_python, String),
    "--control-norm-eps" => real_option(:control_norm_eps),
    "--gain-bound" => real_option(:gain_bound),
    "--control-covariance-weight" => real_option(:control_covariance_weight),
    "--state-covariance-weight" => real_option(:state_covariance_weight),
    "--mass-flow-smoothing" => real_option(:mass_flow_smoothing),
    "--model-mass-flow-smoothing" => real_option(:model_mass_flow_smoothing),
    "--lifting-start" => (:lifting_start, Symbol),
    "--bound-push" => real_option(:bound_push),
    "--eigenvalue-smoothing" => real_option(:eigenvalue_smoothing),
    "--spectral-radius-floor" => real_option(:spectral_radius_floor),
    "--terminal-margin-floor" => real_option(:terminal_margin_floor),
    "--chance-formulation" => (:chance_formulation, Symbol),
    "--objective" => (:objective, Symbol),
    "--chance-slack-floor" => real_option(:chance_slack_floor),
    "--tol" => real_option(:tol),
    "--acceptable-tol" => real_option(:acceptable_tol),
    "--max-iter" => (:max_iter, v -> parse(Int, v)),
    "--alpha-min-frac" => real_option(:alpha_min_frac),
    "--barrier-mu-init" => real_option(:barrier_mu_init),
    "--barrier-mu-min" => real_option(:barrier_mu_min),
    "--barrier-rule" => (:barrier_rule, Symbol),
    "--constr-mult-init-max" => real_option(:constr_mult_init_max),
    "--max-restoration" => (:max_consecutive_restoration, v -> v == "none" ? nothing : parse(Int, v)),
    "--condensed-relaxation" => real_option(:condensed_relaxation),
)

"Validated options of every requested case, or `nothing` after printing the help."
function parse_cli(args)
    settings = Dict{Symbol,Any}()
    cases = ["lyapunov_l1_to_l2"]
    for arg in args
        key, value = occursin('=', arg) ? String.(split(strip(arg), '='; limit=2)) : (strip(arg), "")
        if key in ("--help", "-h")
            print(CLI_HELP)
            return nothing
        elseif key == "--case" && !isempty(value)
            cases = value == "all" ? collect(CASE_IDS) : value == "cr3bp" ? collect(CR3BP_CASE_IDS) : [value]
        elseif key == "--quiet"
            settings[:verbose] = false
        elseif key == "--no-validation"
            settings[:validate_after_solve] = false
        elseif key == "--feasibility"
            settings[:feasibility_only] = true
        elseif key == "--norm-epigraph"
            settings[:norm_epigraph] = true
        elseif key == "--sigma-command-lifting"
            settings[:sigma_command_lifting] = true
        elseif haskey(CLI_OPTIONS, key) && !isempty(value)
            field, parser = CLI_OPTIONS[key]
            settings[field] = parser(value)
        else
            throw(ArgumentError("unknown or incomplete argument $arg\n$CLI_HELP"))
        end
    end
    return [validate_options(Options(; case_id = case, settings...)) for case in cases]
end

"""
Solve and save every requested case. Returns `true` only when every run meets
the reported solver convergence and retained restoration requirements.
"""
function main(args = ARGS)
    runs = parse_cli(args)
    runs === nothing && return true
    check_validation_python(runs)
    succeeded = true
    for o in runs
        result = solve_problem(o)
        save_results(result)
        succeeded &= result.report["converged"]
        result = nothing   # device arrays become collectable before the validation
        o.validate_after_solve && validate_solution(o)
    end
    return succeeded
end
