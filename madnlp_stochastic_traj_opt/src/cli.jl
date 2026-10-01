const CLI_HELP = """
Usage: julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl [options]

  --mode=cpu|gpu                    cpu: sparse KKT system + MUMPS (default)
                                    gpu: condensed KKT system + cuDSS on a CUDA device
  --case=lyapunov_l1_to_l2|nrho_l2_to_dro|halo_l2_to_halo_l1|all   (default lyapunov_l1_to_l2)
  --quiet                           no MadNLP iteration log on the terminal
  --help

Every result, log and diagnostic goes to output/<mode>/<case>/.

Problem robustification (JAX reference values in brackets):
  --control-norm-eps=1e-6           eps in ||S||_eps = sqrt(||S||^2 + eps^2), seed and stochastic solves [1e-6]
  --mass-flow-smoothing=D           delta in the sigma-point mass flow T_max sqrt(||e||^2 + delta^2)
                                    (default: --control-norm-eps) [0, the plain ||u||]
  --eigenvalue-smoothing=1e-6       s in p = sqrt(p2/6 + s^2), at most 1e-4 [1e-15]
  --spectral-radius-floor=1e-6      f in sqrt(lambda_max + f^2), at most 1e-4 [1e-7]
  --terminal-margin-floor=1e-5      terminal margin factor diagonal floor, 0 to 1e-4 [1e-4]

MadNLP, stochastic solve:
  --tol=1e-6  --acceptable-tol=TOL  --max-iter=5000  --alpha-min-frac=0.05
  --barrier-mu-init=0.1  --barrier-mu-min=1e-8   (monotone barrier rule)
  --max-restoration=none|0..5       consecutive restoration limit (default none; the report
                                    always checks the <= 5 requirement)
  --condensed-relaxation=R          gpu: bound_relax_factor of the relaxed equalities
                                    (default TOL/10, at most TOL)
"""

real_option(field) = (field, v -> parse(Float64, v))

const CLI_OPTIONS = Dict(
    "--mode" => (:execution_mode, Symbol),
    "--control-norm-eps" => real_option(:control_norm_eps),
    "--mass-flow-smoothing" => real_option(:mass_flow_smoothing),
    "--eigenvalue-smoothing" => real_option(:eigenvalue_smoothing),
    "--spectral-radius-floor" => real_option(:spectral_radius_floor),
    "--terminal-margin-floor" => real_option(:terminal_margin_floor),
    "--tol" => real_option(:tol),
    "--acceptable-tol" => real_option(:acceptable_tol),
    "--max-iter" => (:max_iter, v -> parse(Int, v)),
    "--alpha-min-frac" => real_option(:alpha_min_frac),
    "--barrier-mu-init" => real_option(:barrier_mu_init),
    "--barrier-mu-min" => real_option(:barrier_mu_min),
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
            cases = value == "all" ? collect(CASE_IDS) : [value]
        elseif key == "--quiet"
            settings[:verbose] = false
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
Solve and save every requested case. Returns `true` when every run ended with
`SOLVE_SUCCEEDED` and met the restoration requirement (the stricter reference
acceptance is reported per run as `converged`).
"""
function main(args = ARGS)
    runs = parse_cli(args)
    runs === nothing && return true
    succeeded = true
    for o in runs
        result = solve_problem(o)
        save_results(result)
        succeeded &= result.report["solver_success"] && result.report["restoration_requirement_met"]
    end
    return succeeded
end
