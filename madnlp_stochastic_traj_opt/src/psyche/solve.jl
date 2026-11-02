# One Psyche run, with the stages of the single-phase driver: energy reoptimisation (CPU),
# TVLQR/UT initial guess, stochastic solve on the run's device, literal re-evaluation and
# the result files read by tools/validate_psyche.py.

"Full-arc solution arrays (Mars arcs carry zero feedforward and gains) from the NLP variables `x`."
function psyche_solution(x, blocks, seed::InitialGuess)
    get(v) = variable_values(x, v)
    controlled = blocks.controlled
    n = size(seed.feedforward, 2)
    feedforward, gains = zeros(NU, n), zeros(NU, NP, n)
    feedforward[:, controlled] = feedforward_values(x, blocks)
    gains[:, :, controlled] = get(blocks.gains) .* reshape(seed.gain_scale[controlled], 1, 1, :)
    level = blocks.level === nothing ? 1.0 : first(get(blocks.level))
    margin = blocks.margin === nothing ? zeros(NM) : get(blocks.margin)
    return (; means = get(blocks.means), feedforward, gains, factors = get(blocks.chol), margin, level)
end

"""
Node standard deviations and packed factors of normalized covariances in the diagonal node-scale
convention (std_k and chol(P_k ./ std_k std_k')), as tools/validate_psyche.py reads them.
"""
function diagonal_convention(covariances)
    n = size(covariances, 3) - 1
    std = sqrt.(max.(reduce(hcat, [diag(covariances[:, :, k]) for k in 1:n+1]), 1e-24))
    factor(k) = (P = covariances[:, :, k] ./ (std[:, k] * std[:, k]'); pack_lower(Matrix(cholesky(Symmetric((P + P') / 2)).L)))
    return std, reduce(hcat, [factor(k) for k in 2:n+1]), factor(1)
end

"Normalized node covariances of a solution, the initial one included."
function psyche_covariances(scaling::CovarianceScaling, factors)
    n = size(factors, 2)
    covariances = zeros(NX, NX, n + 1)
    for k in 1:n+1
        F = node_factor(factors, scaling, k)
        covariances[:, :, k] = F * F'
    end
    return covariances
end

"""
Literal re-evaluation of a Psyche point with the factor form of the UT arc map: residuals,
chance values, the weighted cost and its shares, terminal and encounter statistics.
"""
function psyche_evaluate(d::PsycheData, o::Options, u::PsycheUncertainty, params, layout::StochasticLayout,
        scaling::CovarianceScaling, seed::InitialGuess, sol)
    n = d.n
    covariances = psyche_covariances(scaling, sol.factors)
    pc = layout.control_params
    controlled = d.slots .> 0
    mean_defect, covariance_defect = 0.0, 0.0
    chance = zeros(n)
    control, state, control_covariance = 0.0, 0.0, 0.0
    for k in 1:n
        mu, P, T = propagation_arc_factor(params[d.phase[k]], sol.means[:, k], node_factor(sol.factors, scaling, k),
            sol.feedforward[:, k], sol.gains[:, :, k], layout.steps[k])
        predicted = layout.ratio[:, k] .* mu + layout.defect_rhs[:, k]
        mean_defect = max(mean_defect, maximum(abs, sol.means[:, k+1] - predicted))
        L = unpack_lower(sol.factors[:, k], NX)
        covariance_defect = max(covariance_defect, maximum(abs, L * L' - node_scaled_covariance(scaling, k + 1, P)))
        w = layout.weights[k]
        state += w * pc.inv_sigma_sq * tr(covariances[1:NP, 1:NP, k])
        if controlled[k]
            f = SVector{3}(sol.feedforward[:, k])
            chance[k] = feedforward_norm(pc, f) + pc.psi * spectral_radius(SMatrix{3,3}(T), pc.floor, pc.smoothing_sq)
            control += w * feedforward_norm(pc, f)
            control_covariance += w * o.control_covariance_weight * tr(T)
        end
    end
    G = layout.terminal_transform
    target = G * covariances[1:NP, 1:NP, end] * G'
    terminal_bound = o.terminal_covariance == :constraint
    terminal = terminal_bound ? 0.0 : o.terminal_covariance_weight * tr(target) / o.bryson_sigma_factor^2
    G = unpack_lower(sol.margin, NP)
    total = control + state + control_covariance + terminal
    # objective = :chance_sum: the arc-weighted chance left-hand sides are the cost (the terms above are diagnostics)
    chance_cost = sum(layout.weights[k] * chance[k] for k in 1:n if controlled[k])
    si(k) = psyche_si_covariance(u, covariances[:, :, k])
    ca = d.ca_node
    clearance = psyche_clearance(d, u, scaling)
    energy = sum(0.5 * d.thrust_max_n^2 * layout.steps[k] * d.time_scales[d.phase[k]] * sum(abs2, sol.feedforward[:, k])
                 for k in 1:n)
    scaled_gains = sol.gains[:, :, controlled] ./ reshape(seed.gain_scale[controlled], 1, 1, :)
    return Dict{String,Any}(
        "objective" => o.objective == :chance_sum ? chance_cost : total, "objective_chance_sum" => chance_cost,
        "objective_mean_control" => control, "objective_state_covariance" => state,
        "objective_control_covariance" => control_covariance,
        "objective_share_mean_control" => control / total, "objective_share_state_covariance" => state / total,
        "objective_share_control_covariance" => control_covariance / total,
        "objective_terminal_covariance" => terminal, "objective_share_terminal_covariance" => terminal / total,
        "max_mean_defect" => mean_defect, "max_scaled_covariance_defect" => covariance_defect,
        "max_boundary_residual" => maximum(abs(sol.means[t.i, t.node] - t.value) for t in layout.boundary),
        "max_terminal_constraint_violation" => terminal_bound ? maximum(abs, sol.level * I - target - G * G') :
            max(0.0, eigmax(Symmetric(target)) - 1),
        "terminal_dispersion_level" => sol.level,
        "terminal_covariance_max_eigenvalue" => eigmax(Symmetric(target)),
        "terminal_std_si" => sqrt.(diag(si(n + 1))),
        "max_thrust_budget" => maximum(chance[controlled]),
        "max_control_chance_violation" => max(0.0, maximum(chance[controlled]) - 1),
        "active_chance_constraints" => count(>(1 - 1e-4), chance[controlled]),
        "chance_by_arc" => chance,
        "minimum_mean_mass_kg" => minimum(sol.means[7, :] .* d.node_scales[7, :]),
        "propellant_kg" => (sol.means[7, 1] - sol.means[7, end]) * d.node_scales[7, 1],
        "nominal_command_energy_n2_s" => energy,
        "max_scaled_gain" => maximum(abs, scaled_gains),
        "encounter" => encounter_report(sol.means[1:NP, ca] .* d.node_scales[1:NP, ca], si(ca)[1:NP, 1:NP],
            clearance.mu, clearance.rho, clearance.chi))
end

function psyche_save_toml(path, report)
    open(path, "w") do io
        TOML.print(io, toml_ready(report); sorted = true)
    end
end

function _solve_psyche(o::Options, backend)
    directory = mkpath(output_directory(o))
    started = time()
    stage = "mission inputs"
    tag = "[psyche]"
    report = Dict{String,Any}("case" => "psyche", "execution_mode" => String(o.execution_mode),
        "implementation" => "native Julia/ExaModels (shared stochastic NLP)", "converged" => false,
        "solver_success" => false, "substeps_per_interval" => 1,
        "mars_events" => "fixed at the energy reoptimisation epochs", "ballistic_mars" => true,
        "flyby_risk" => PSYCHE_FLYBY_RISK, "minimum_mars_altitude_m" => PSYCHE_MINIMUM_MARS_ALTITUDE_M,
        "initial_position_std_km" => o.initial_position_std_km, "initial_mass_std_kg" => o.initial_mass_std_kg,
        "terminal_position_std_km" => o.psyche_terminal_position_std_km,
        "terminal_velocity_std_km_s" => o.psyche_terminal_velocity_std_km_s,
        "control_chance_covariance" => "executed: commanded plus execution",
        "seed_gain" => String(o.psyche_seed_gain), "terminal_covariance" => String(o.terminal_covariance),
        "control_parameterization" => String(o.control_parameterization))
    try
        d = load_psyche_data(o.psyche_input_file)
        u = psyche_uncertainty(d, o)
        params = psyche_phase_params(d, o, u)
        # Throttle controls: the energy reoptimisation and the seed use the model's exact parameters as well
        # (mass flow T_max Gamma, no smoothing, no Gates jitter), as in the deterministic optimisation.
        nominal_params = o.control_parameterization == :throttle ? model_params(o, params) : params
        merge!(report, Dict{String,Any}("input_sha256" => d.sha256, "n_intervals" => d.n, "n_arcs" => d.n,
            "powered_control_arcs" => count(>(0), d.slots), "phase_mesh_counts" => d.mesh_counts,
            "mesh_refinement" => d.mesh_refinement, "cruise_subdivision" => d.cruise_subdivision,
            "reference_path" => d.reference_path, "initial_std_si" => sqrt.(diag(u.initial_covariance_si)),
            "terminal_std_limits_si" => u.terminal_std_si,
            "initialisation" => "energy reoptimisation -> duration-weighted Bryson TVLQR -> UT covariance -> terminal margin"))
        log_line(@sprintf("%s %d intervals (%d powered), UT dimension %d, mode %s", tag, d.n, count(>(0), d.slots),
            UT_DIMENSION, o.execution_mode))

        stage = "energy reoptimisation"
        log_line("$tag $(o.psyche_nominal_objective) reoptimisation (RK9, free Mars epochs, MadNLP/MUMPS on the CPU) ...")
        nominal_seconds = @elapsed ref, nominal = psyche_reoptimize(d, o, nominal_params;
            output_file = joinpath(directory, "psyche_madnlp_nominal.log"))
        report["nominal"] = nominal
        NPZ.npzwrite(joinpath(directory, "psyche_nominal.npz"),
            Dict("means" => ref.means, "feedforward" => ref.feedforward, "days" => ref.days))
        log_line(@sprintf("%s reference: %s in %d iterations, energy %.6e N^2 s, Mars days %.6f + %.6f (%.1f s)", tag,
            nominal["status"], nominal["iterations"], nominal["energy_n2_s"], ref.days..., nominal_seconds))

        stage = "initial guess"
        seed_seconds = @elapsed seed, scaling = psyche_initial_guess(d, o, u, nominal_params, ref)
        seed_std, seed_factors, seed_initial = diagonal_convention(seed.covariances)
        NPZ.npzwrite(joinpath(directory, "psyche_seed.npz"), Dict("means" => seed.means, "feedforward" => seed.feedforward,
            "days" => ref.days, "gains" => seed.gains, "gain_scale" => seed.gain_scale, "covariances" => seed.covariances,
            "covariance_std" => seed_std, "factors" => seed_factors, "initial_factor" => seed_initial,
            "margin" => seed.terminal_margin))

        stage = "stochastic model"
        # The nominal and seed solves above keep `params`; polar and throttle controls change the model's (`model_params`).
        mparams = model_params(o, params)
        build_seconds = @elapsed blocks = psyche_stochastic_model(d, o, u, mparams, ref, seed, scaling, backend)
        model = solver_model(o, blocks.model)
        x0 = Array(model.meta.x0)
        seed_violation = original_violation(model, model.meta.x0)
        seed_report = psyche_evaluate(d, o, u, mparams, blocks.layout, scaling, seed, psyche_solution(x0, blocks, seed))
        log_line(@sprintf("%s model: %d variables, %d constraints; seed: max violation %.3e, objective %.10f", tag,
            model.meta.nvar, model.meta.ncon, seed_violation, seed_report["objective"]))

        stage = "stochastic solve"
        log_line("$tag stochastic solve ...")
        settings = stochastic_solver_options(o; output_file = joinpath(directory, "psyche_madnlp_stochastic.log"))
        guard = settings.intermediate_callback
        solve_seconds = @elapsed stats, solver = run_madnlp(model, settings)

        x = Array(stats.solution)
        sol = psyche_solution(x, blocks, seed)
        evaluation = psyche_evaluate(d, o, u, mparams, blocks.layout, scaling, seed, sol)
        events = psyche_event_report(d, sol.means, ref.days)
        values = Array(NLPModels.cons(model, stats.solution))
        success = stats.status == MadNLP.SOLVE_SUCCEEDED
        restoration_ok = !guard.limit_exceeded && guard.max_consecutive <= MAX_RESTORATION_RUN
        domain = events["strict_hyperbolic_minimum"]
        merge!(report, evaluation, Dict{String,Any}(
            "madnlp_status" => string(stats.status), "iterations" => stats.iter, "solver_success" => success,
            "model_objective" => stats.objective + blocks.objective_constant,
            "objective_units" => o.objective == :terminal_dispersion ? "level t of G P_N G' <= t I (G: terminal target)" :
                "dimensionless, arc-duration-weighted stochastic cost",
            "restoration_requirement_met" => restoration_ok, "restoration_iterations" => guard.restoration_iterations,
            "longest_restoration_run" => guard.max_consecutive, "blocked_restoration_request" => String(guard.blocked_phase),
            "hyperbolic_minimum_valid" => domain, "converged" => success && restoration_ok && domain,
            "max_constraint_violation" => original_violation(model, stats.solution),
            "bplane_clearance_row" => values[end], "events" => events,
            "nvar" => model.meta.nvar, "ncon" => model.meta.ncon, "final_kkt" => kkt_measures(solver),
            "settings" => solver_settings_report(o, solver),
            "seed" => Dict{String,Any}("max_constraint_violation" => seed_violation, "objective" => seed_report["objective"],
                "objective_share_mean_control" => seed_report["objective_share_mean_control"],
                "objective_share_state_covariance" => seed_report["objective_share_state_covariance"],
                "objective_share_control_covariance" => seed_report["objective_share_control_covariance"]),
            "seconds" => Dict{String,Any}("nominal" => nominal_seconds, "seed" => seed_seconds,
                "build" => build_seconds, "solve" => solve_seconds),
            "julia_threads" => Threads.nthreads()))
        report["settings"]["psyche_nominal_barrier_mu_init"] = o.psyche_nominal_barrier_mu_init
        covariances = psyche_covariances(scaling, sol.factors)
        solution_std, solution_factors, _ = diagonal_convention(covariances)
        NPZ.npzwrite(joinpath(directory, "psyche_solution.npz"), Dict("x" => x, "means" => sol.means,
            "feedforward" => sol.feedforward, "days" => ref.days, "gains" => sol.gains, "gain_scale" => seed.gain_scale,
            "factors" => solution_factors, "margin" => sol.margin, "covariances" => covariances,
            "covariance_std" => solution_std, "constraints" => values, "node_scale_full" => Int(scaling.full)))
        print_report(tag, report)
        log_line(@sprintf("  cost shares: mean control %.2f %%, state covariance %.2f %%, control covariance %.2f %%, terminal %.2f %%",
            100report["objective_share_mean_control"], 100report["objective_share_state_covariance"],
            100report["objective_share_control_covariance"], 100report["objective_share_terminal_covariance"]))
        log_line("  hyperbolic closest-approach minimum: ", domain ? "yes" : "no",
            @sprintf("; B-plane clearance %.4f", values[end]))
    catch exception
        report["converged"] = false
        report["failed_stage"] = stage
        report["error"] = sprint(showerror, exception, catch_backtrace())
        get!(report, "madnlp_status", "FAILED_BEFORE_RESULT")
        log_line("$tag failed during ", stage, ": ", sprint(showerror, exception))
    finally
        report["total_wall_seconds"] = time() - started
        psyche_save_toml(joinpath(directory, "psyche_diagnostics.toml"), report)
    end
    return (; report, options = o)
end
