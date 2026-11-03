const CLI_HELP = """
Usage: julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl [options]

  --case=lyapunov_l1_to_l2|nrho_l2_to_dro|halo_l2_to_halo_l1|earth_mars|psyche|cr3bp|all
                                    (default lyapunov_l1_to_l2; cr3bp: the three CR3BP cases)
  --objective=standard|chance_sum   standard (step 3, default): tracking cost with the terminal covariance in
                                    the cost and the thrust chance rows; chance_sum (step 4): time-weighted sum
                                    of the thrust chance bounds (slack epigraph), hard terminal covariance bound
                                    and Bryson seed weights (results in output/<mode>/<case>_chance_sum/)
  --mode=cpu|gpu                    cpu: model and MadNLP (sparse KKT system + MUMPS) on the CPU (default)
                                    gpu: the model evaluated on a CUDA device, MadNLP and MUMPS on the CPU
  --max-iter=N                      MadNLP iteration budget (default 5000; chance_sum 10000)
  --quiet                           no MadNLP iteration log on the terminal
  --no-validation                   skip the Monte Carlo validation after the solve
  --validation-python=PATH          Python with JAX for the validation
                                    (default: \$MADNLP_VALIDATION_PYTHON, else \$PSYCHE_PYTHON,
                                    else ~/anaconda3/envs/project/bin/python, else python3)
  --help
"""

const CLI_OPTIONS = Dict(
    "--objective" => (:objective, Symbol),
    "--mode" => (:execution_mode, Symbol),
    "--max-iter" => (:max_iter, v -> parse(Int, v)),
    "--validation-python" => (:validation_python, String),
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
