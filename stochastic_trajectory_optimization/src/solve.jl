# Orchestration of one run: fuel-optimal reoptimised reference, TVLQR/UT initial guess, stochastic solve.

log_line(args...) = (println(args...); flush(stdout))

function _solve_problem(o::Options, backend)
    case = case_data(o.case_id)
    normalization = build_normalization(case, o)
    p = ut_params(case, o, normalization)
    directory = mkpath(output_directory(o))
    tag = "[$(case.id)]"
    source = load_reference(case, o)
    n = n_arcs(source)
    log_line(@sprintf("%s %d arcs, UT dimension %d (%d outer, %d nested sigma points), mode %s",
        tag, n, p.d, 2p.d + 1, 6(2p.d + 1), o.execution_mode))

    log_line("$tag fuel-optimal reference (throttle and direction, Dopri8 shooting, MadNLP/MUMPS on the CPU) ...")
    nominal_seconds = @elapsed ref, nominal = reoptimize_reference(case, o, source;
        output_file = joinpath(directory, case.id * "_madnlp_nominal.log"))
    log_line(@sprintf("%s reference: %s in %d iterations, objective %.10f, violation %.2e (%.1f s)",
        tag, nominal["status"], nominal["iterations"], nominal["objective"], nominal["max_constraint_violation"],
        nominal_seconds))

    seed_seconds = @elapsed seed = build_initial_guess(case, o, normalization, p, ref)
    scaling = covariance_scaling(seed, normalization)
    build_seconds = @elapsed blocks = build_stochastic_model(case, o, ref, seed, scaling, p, backend)
    model = solver_model(o, blocks.model)
    seed_violation = original_violation(model, model.meta.x0)
    seed_report = evaluate_solution(case, o, ref, extract_solution(model.meta.x0, blocks, scaling, seed, n), scaling, p)
    log_line(@sprintf("%s model: %d variables, %d constraints; TVLQR seed: max violation %.3e, objective %.10f",
        tag, model.meta.nvar, model.meta.ncon, seed_violation, seed_report["objective"]))

    log_line("$tag stochastic solve ...")
    settings = stochastic_solver_options(o; output_file = joinpath(directory, case.id * "_madnlp_stochastic.log"))
    guard = settings.intermediate_callback
    solve_seconds = @elapsed stats, solver = run_madnlp(model, settings)

    solution = extract_solution(stats.solution, blocks, scaling, seed, n)
    report = evaluate_solution(case, o, ref, solution, scaling, p)
    violation = max(original_violation(model, stats.solution), report["absolute_feasibility"])
    solver_success = stats.status == MadNLP.SOLVE_SUCCEEDED
    restoration_ok = !guard.limit_exceeded && guard.max_consecutive <= MAX_RESTORATION_RUN
    merge!(report, Dict{String,Any}(
        "case" => case.id, "execution_mode" => String(o.execution_mode),
        "n_arcs" => n, "nvar" => model.meta.nvar, "ncon" => model.meta.ncon,
        "madnlp_status" => string(stats.status), "iterations" => stats.iter,
        "solver_success" => solver_success,
        "restoration_requirement_met" => restoration_ok,
        "restoration_iterations" => guard.restoration_iterations,
        "longest_restoration_run" => guard.max_consecutive,
        "blocked_restoration_request" => String(guard.blocked_phase),
        "max_constraint_violation" => violation,
        "original_residual_acceptance_check" => false,
        "acceptance_policy" => "MadNLP solver success and retained restoration requirement; original residuals are diagnostic only",
        "converged" => solver_success && restoration_ok,
        "model_objective" => stats.objective + blocks.objective_constant,
        "final_kkt" => kkt_measures(solver),
        "settings" => solver_settings_report(o, solver),
        "seed" => Dict{String,Any}("max_constraint_violation" => seed_violation, "objective" => seed_report["objective"]),
        "nominal" => nominal,
        "seconds" => Dict{String,Any}("nominal" => nominal_seconds, "seed" => seed_seconds, "build" => build_seconds,
            "solve" => solve_seconds),
        "julia_threads" => Threads.nthreads()))
    print_report(tag, report)
    return (; options = o, case, normalization, params = p, reference = ref, seed, scaling, solution, report)
end

function solver_settings_report(o::Options, solver)
    settings = Dict{String,Any}("objective" => string(o.objective), "tol" => o.tol, "acceptable_tol" => o.acceptable_tol,
        "max_iter" => o.max_iter, "alpha_min_frac" => ALPHA_MIN_FRAC, "barrier_mu_init" => BARRIER_MU_INIT,
        "barrier_mu_min" => BARRIER_MU_MIN, "constr_mult_init_max" => CONSTR_MULT_INIT_MAX,
        "control_covariance_weight" => o.control_covariance_weight, "state_covariance_weight" => o.state_covariance_weight,
        "seed_weights" => collect(seed_weights(o)),
        "eigenvalue_smoothing" => EIGENVALUE_SMOOTHING, "spectral_radius_floor" => SPECTRAL_RADIUS_FLOOR,
        "terminal_margin_floor" => TERMINAL_MARGIN_FLOOR, "terminal_covariance_weight" => TERMINAL_COVARIANCE_WEIGHT,
        "chance_slack_floor" => CHANCE_SLACK_FLOOR, "kkt_system" => string(LINEAR_SYSTEM.kkt_system),
        # what verification/__main__.py replays: the reference's eps and smoothing (control_norm_eps,
        # mass_flow_smoothing, gain_bound) and the model's exact norm, sigma-point mass flow and Gates factors
        "control_norm_eps" => CONTROL_NORM_EPS, "mass_flow_smoothing" => CONTROL_NORM_EPS, "gain_bound" => Inf,
        "model_norm_smoothing" => 0.0, "model_mass_flow_smoothing" => MASS_FLOW_SMOOTHING,
        "model_mass_flow_smoothing_feedback_only" => true, "model_gates_jitter" => 0.0)
    # Effective values, including the MadNLP defaults of options that are not passed.
    settings["bound_push"] = solver.opt.bound_push
    settings["bound_relax_factor"] = solver.opt.bound_relax_factor
    settings["inertia_correction"] = string(nameof(typeof(solver.inertia_corrector)))
    return settings
end

function print_report(tag, r)
    kkt = r["final_kkt"]
    yes(flag) = flag ? "yes" : "no"
    log_line(@sprintf("%s MadNLP %s after %d iterations (%.1f s)", tag, r["madnlp_status"], r["iterations"],
        r["seconds"]["solve"]))
    log_line(@sprintf("  objective %.10f   original max constraint violation %.3e (diagnostic only)",
        r["objective"], r["max_constraint_violation"]))
    log_line(@sprintf("  unscaled: dual infeasibility %.3e, constraint violation %.3e, complementarity %.3e",
        kkt["dual_infeasibility_unscaled"], kkt["constraint_violation_unscaled"], kkt["complementarity_unscaled"]))
    log_line(@sprintf("  restoration: %d iterations, longest run %d (requirement <= %d): %s",
        r["restoration_iterations"], r["longest_restoration_run"], MAX_RESTORATION_RUN,
        r["restoration_requirement_met"] ? "met" : "violated"))
    log_line("  converged (solver success and restoration requirement): ", yes(r["converged"]))
end

"""
    solve_problem(options) -> result

Fuel-optimal reference, TVLQR/UT seed and the stochastic MadNLP solve for one
case, execution mode and problem. `result.report` holds the solver status, the final KKT
measures and independently recomputed residuals; `save_results` writes them.
"""
function solve_problem(o::Options = Options())
    validate_options(o)
    backend = execution_backend(o)
    return Base.invokelatest(o.case_id == "psyche" ? _solve_psyche : _solve_problem, o, backend)
end
