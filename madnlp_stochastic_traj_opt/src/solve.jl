function _solve_problem(o::Options, backend)
    data = system_matrices(o)
    Qc, Lc = calibrate_continuous_diffusion(data.Ac, data.G*data.G', o.dt)
    shark_error = validate_shark(o, data.Ac, Qc, Lc)
    @printf("ShARK process covariance error: %.3e\n", shark_error)
    println("Solving deterministic minimum-energy trajectory ...")
    # Compute the same accurately constrained energy trajectory for every backend.
    # Gain initialization and seed covariance propagation also run on CPU.
    nominal_blocks = build_nominal_model(o, nothing)
    nominal_settings = solver_options(o;nominal=true)
    nominal_stats = run_madnlp(nominal_blocks.model,o,nominal_settings)
    nominal_settings.intermediate_callback.blocked_phase == :none ||
        error("nominal solver would exceed the consecutive restoration limit; stopped")
    nominal_stats.status == MadNLP.SOLVE_SUCCEEDED || error("nominal energy solve did not converge: $(nominal_stats.status)")
    nominal_violation = original_violation(nominal_blocks.model, nominal_stats.solution)
    nominal_violation <= 10o.feasibility_tol || error("nominal trajectory is infeasible: $nominal_violation")
    nominal = (; means=Array(ExaModels.solution(nominal_stats,nominal_blocks.means)),
        feedforward=Array(ExaModels.solution(nominal_stats,nominal_blocks.feedforward)))
    seed = build_seed(o, data.A, data.B, Lc, nominal)
    println("Building 25-point UT stochastic model ...")
    build_seconds = @elapsed blocks = build_stochastic_model(o, Lc, seed, backend)
    seed_sol = merge(seed, (; state_covariances=cat((let L=unpack_lower(seed.cholesky_factor[:,k]); L*L' end for k in 1:o.n_arcs+1)...;dims=3)))
    seed_diagnostics = diagnostics(o, Lc, seed_sol)
    @assert seed_diagnostics.mean_defect <= 1e-12
    @assert seed_diagnostics.covariance_defect <= 1e-12
    @printf("Model: %d variables, %d constraints; build %.2f s\n", blocks.model.meta.nvar, blocks.model.meta.ncon, build_seconds)
    @printf("Seed mean/covariance defects: %.3e / %.3e\n", seed_diagnostics.mean_defect, seed_diagnostics.covariance_defect)
    seed_violation = original_violation(blocks.model, blocks.model.meta.x0)
    @printf("Seed maximum constraint violation: %.3e; initial gain range: [%.3e, %.3e]\n",
        seed_violation,minimum(seed.gains),maximum(seed.gains))
    println("Solving stochastic covariance steering ...")
    solve_seconds = @elapsed run = solve_stochastic_model(blocks.model,o)
    stats, guard = run.stats, run.guard
    sol = extract_solution(stats, blocks)
    diag = diagnostics(o, Lc, sol)
    violation = original_violation(blocks.model, stats.solution)
    report = Dict{String,Any}(
        "julia_version"=>string(VERSION), "julia_optimization_level"=>Int(Base.JLOptions().opt_level),
        "execution_mode"=>String(o.execution_mode), "solver_status"=>string(stats.status),
        "restoration_enabled"=>o.max_consecutive_restoration > 0,
        "restoration_iterations"=>guard.restoration_iterations,
        "max_consecutive_restoration"=>guard.max_consecutive,
        "restoration_limit"=>o.max_consecutive_restoration,
        "restoration_limit_exceeded"=>guard.limit_exceeded,
        "blocked_restoration_request"=>String(guard.blocked_phase),
        "converged"=>stats.status == MadNLP.SOLVE_SUCCEEDED && !guard.limit_exceeded &&
            max(violation,diag.independent_violation) <= 10o.feasibility_tol,
        "iterations"=>run.total_iterations, "solver_stages"=>run.stages,
        "iteration_limit"=>o.max_iter, "nominal_backend"=>"cpu",
        "dual_infeasibility"=>stats.dual_feas,
        "objective"=>diag.objective, "max_constraint_violation"=>violation,
        "feasible"=>violation <= 10o.feasibility_tol,
        "feasibility_target"=>o.feasibility_tol, "acceptance_tolerance"=>10o.feasibility_tol,
        "mean_matching_defect"=>diag.mean_defect, "covariance_matching_defect"=>diag.covariance_defect,
        "independent_constraint_violation"=>diag.independent_violation,
        "terminal_margin_residual"=>diag.terminal_residual,
        "terminal_headroom_min_eigenvalue"=>diag.terminal_headroom_min_eigenvalue,
        "shark_process_covariance_error"=>shark_error, "smoothing_radius_bias"=>diag.smoothing_bias,
        "spectral_eigenvalue_smoothing"=>o.spectral_eigenvalue_smoothing,
        "initial_barrier"=>o.initial_barrier,
        "alpha_min_frac"=>o.alpha_min_frac,
        "nominal_restoration_iterations"=>nominal_settings.intermediate_callback.restoration_iterations,
        "nominal_max_consecutive_restoration"=>nominal_settings.intermediate_callback.max_consecutive,
        "hybrid_gamma"=>o.execution_mode == :gpu_hybrid ? o.hybrid_gamma : 0.0,
        "seed_mean_matching_defect"=>seed_diagnostics.mean_defect,
        "tvlqr_control_weight_scale"=>o.tvlqr_control_weight_scale,
        "initial_gain_max_abs"=>o.initial_gain_max_abs === nothing ? "unscaled" : o.initial_gain_max_abs,
        "initial_gain_uniform"=>o.initial_gain_uniform === nothing ? "disabled" : o.initial_gain_uniform,
        "seed_min_gain"=>minimum(seed.gains), "seed_max_gain"=>maximum(seed.gains),
        "seed_max_abs_gain"=>maximum(abs,seed.gains),
        "seed_covariance_matching_defect"=>seed_diagnostics.covariance_defect,
        "seed_max_constraint_violation"=>seed_violation,
        "nvar"=>blocks.model.meta.nvar, "ncon"=>blocks.model.meta.ncon,
        "build_seconds"=>build_seconds, "solve_seconds"=>solve_seconds,
        "terminal_mean"=>sol.means[:,end], "terminal_variances"=>diagind_values(sol.state_covariances[:,:,end]),
        "solver_tolerance"=>o.tol, "equality_relaxation"=>o.execution_mode == :gpu_condensed ? o.condensed_relaxation : 0.0)
    # Store formulation data so cross-language verification also works for
    # customized horizons, covariances, costs, and chance levels.
    problem_fields = (:dt,:tf,:n_arcs,:x0_mean,:x0_covariance,:xf_mean,:xf_covariance,
        :a1,:b1,:a2,:b2,:path_confidence,:control_confidence,:u_max,:Q,:R,
        :scaling_parameter,:spectral_eigenvalue_smoothing,:terminal_margin_floor,:control_norm_epsilon)
    report["problem"] = Dict(String(name) => (let value=getproperty(o,name)
        value isa AbstractMatrix ? [collect(row) for row in eachrow(value)] : value
    end) for name in problem_fields)
    println("Status: ", report["solver_status"])
    guard.blocked_phase == :none || println("Stopped before exceeding the consecutive ",guard.blocked_phase," restoration limit.")
    @printf("Objective %.10f; original constraint violation %.3e; feasible %s\n", diag.objective, violation, string(report["feasible"]))
    return (; options=o, solution=merge(sol,diag), report, stats, nominal, seed, Qc, Lc)
end

diagind_values(P) = diag(P)

"""
Solve the nominal energy transfer and stochastic covariance-steering problem.

Returns options, solution arrays, seed, solver statistics, and an independently
checked report without writing files. Check `result.report["converged"]`
before accepting the stochastic result; use `save_results` for output.
"""
function solve_problem(o::Options=Options())
    validate_options(o)
    backend = execution_backend(o)
    return Base.invokelatest(_solve_problem, o, backend)
end
