"Parse command-line options without loading a solver or writing output."
function parse_cli_options(args)
    settings = Dict{Symbol,Any}()
    for arg in args
        if startswith(arg,"--mode=")
            settings[:execution_mode] = Symbol(split(arg,'=';limit=2)[2])
        elseif arg == "--no-plots"
            settings[:make_plots] = false
        elseif arg == "--no-monte-carlo"
            settings[:run_monte_carlo] = false
        elseif startswith(arg,"--output=")
            settings[:output_dir] = split(arg,'=';limit=2)[2]
        elseif startswith(arg,"--samples=")
            settings[:monte_carlo_samples] = parse(Int,split(arg,'=';limit=2)[2])
        elseif startswith(arg,"--max-iter=")
            settings[:max_iter] = parse(Int,split(arg,'=';limit=2)[2])
        elseif startswith(arg,"--max-restoration=")
            settings[:max_consecutive_restoration] = parse(Int,split(arg,'=';limit=2)[2])
        elseif startswith(arg,"--eigenvalue-smoothing=")
            settings[:spectral_eigenvalue_smoothing] = parse(Float64,split(arg,'=';limit=2)[2])
        elseif startswith(arg,"--terminal-factor-floor=")
            settings[:terminal_margin_floor] = parse(Float64,split(arg,'=';limit=2)[2])
        elseif startswith(arg,"--initial-gain-max-abs=")
            value = split(arg,'=';limit=2)[2]
            settings[:initial_gain_max_abs] = value == "none" ? nothing : parse(Float64,value)
        elseif startswith(arg,"--initial-gain-uniform=")
            value = split(arg,'=';limit=2)[2]
            settings[:initial_gain_uniform] = value == "none" ? nothing : parse(Float64,value)
        elseif startswith(arg,"--tvlqr-control-weight-scale=")
            settings[:tvlqr_control_weight_scale] = parse(Float64,split(arg,'=';limit=2)[2])
        elseif startswith(arg,"--condensed-relaxation=")
            settings[:condensed_relaxation] = parse(Float64,split(arg,'=';limit=2)[2])
        elseif startswith(arg,"--tol=")
            settings[:tol] = parse(Float64,split(arg,'=';limit=2)[2])
        elseif startswith(arg,"--alpha-min-frac=")
            settings[:alpha_min_frac] = parse(Float64,split(arg,'=';limit=2)[2])
        elseif arg == "--quiet"
            settings[:print_level] = 0
        elseif arg == "--verbose"
            settings[:print_level] = 2
        elseif arg in ("--help","-h")
            print(CLI_HELP)
            return nothing
        else
            throw(ArgumentError("unknown argument: $arg"))
        end
    end
    return validate_options(Options(;settings...))
end

"Run optimization and the requested output/verification steps."
function main(args=ARGS)
    options = parse_cli_options(args)
    options === nothing && return nothing
    result = solve_problem(options)
    save_results(result)
    result.report["converged"] || error("solve did not meet convergence and original feasibility requirements; see saved diagnostics")
    return result
end

const CLI_HELP = """
Usage: julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/run.jl [options]

Execution and output:
  --mode=cpu|gpu_condensed|gpu_hybrid  Default: cpu
  --output=PATH                      Default: project/output/<mode>
  --no-plots                         Disable plot generation
  --no-monte-carlo                   Disable Monte Carlo verification
  --samples=8192                     Power-of-two Monte Carlo sample count
  --quiet | --verbose                Solver log verbosity

Solver settings:
  --tol=1e-8                         KKT tolerance
  --max-iter=10000                   Stochastic iteration budget
  --max-restoration=2                Maximum consecutive restoration steps (0–2)
  --alpha-min-frac=0.05              Line-search threshold, all modes
  --condensed-relaxation=1e-9         Condensed GPU bound-relaxation factor
  --eigenvalue-smoothing=1e-5        Control-covariance eigenvalue smoothing
  --terminal-factor-floor=1e-7       Terminal factor diagonal floor

Initialization:
  --initial-gain-uniform=1e-5|none    Uniform entries; none selects TVLQR
  --initial-gain-max-abs=VALUE|none   Optional TVLQR rescaling (default: none)
  --tvlqr-control-weight-scale=VALUE Optional TVLQR penalty multiplier

Uniform initialization and TVLQR rescaling are mutually exclusive.
  --help                            Show this message
"""
