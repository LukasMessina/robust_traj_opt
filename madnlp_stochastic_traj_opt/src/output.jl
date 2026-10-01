# Result files in output/<mode>/<case>/. `<case>.npz` uses the JAX output key names
# and axis conventions wherever they apply and carries everything tools/validate.py
# needs to re-evaluate the solution with the reference code; `<case>_diagnostics.toml`
# holds the report. MadNLP writes its logs to the same directory during the solve.

"Write `<case>.npz` and `<case>_diagnostics.toml`; returns the directory."
function save_results(result)
    o, case, sol, report = result.options, result.case, result.solution, result.report
    ref, scaling, seed, n = result.reference, result.scaling, result.seed, n_arcs(result.reference)
    directory = mkpath(output_directory(o))
    scale = Vector(result.normalization.scale)
    to_python(A::AbstractArray{<:Any,3}) = permutedims(A, (3, 1, 2))   # (N, rows, cols)
    arrays = Dict{String,Any}(
        # run identification (the case id is the file name)
        "n_arcs" => n, "objective" => report["objective"],
        "solver_success" => Int(report["solver_success"]),
        "restoration_requirement_met" => Int(report["restoration_requirement_met"]),
        "eigenvalue_smoothing" => o.eigenvalue_smoothing, "terminal_margin_floor" => o.terminal_margin_floor,
        "spectral_radius_floor" => o.spectral_radius_floor, "control_norm_eps" => o.control_norm_eps,
        "mass_flow_smoothing" => o.mass_flow_smoothing,
        # time grid and energy-optimal reference
        "node_times_nd" => ref.node_times, "node_days" => ref.node_times .* case.time_unit ./ 86400.0,
        "steps_nd" => ref.steps, "ref_states" => ref.states, "ref_controls" => ref.controls,
        # solution (JAX names)
        "means" => sol.means, "feedforward" => sol.feedforward,
        "feedforward_n" => sol.feedforward .* case.tmax .* case.thrust_unit,
        "gains_normalized" => to_python(sol.gains),
        "gains" => to_python(case.tmax .* sol.gains ./ reshape(scale[1:NP], 1, :, 1)),
        "radius_normalized" => sol.radius,
        "covariance_normalized" => to_python(sol.covariances),
        "control_covariance_normalized" => to_python(sol.control_covariances),
        "normalization_scale_nd" => scale,
        # NLP variables and their fixed scalings
        "cholesky_factor" => permutedims(sol.cholesky_factor, (2, 1)),
        "terminal_margin" => sol.terminal_margin,
        "node_std" => scaling.std,
        "covariance_scale_variances" => permutedims(scaling.variances, (2, 1)),
        "gain_scale" => seed.gain_scale,
        # initial guess
        "seed_gains_normalized" => to_python(seed.gains),
        "seed_covariances" => to_python(seed.covariances))
    NPZ.npzwrite(joinpath(directory, case.id * ".npz"), arrays)
    open(joinpath(directory, case.id * "_diagnostics.toml"), "w") do io
        TOML.print(io, toml_ready(report); sorted = true)
    end
    log_line("[$(case.id)] results written to $directory")
    return directory
end

toml_ready(d::AbstractDict) = Dict{String,Any}(string(k) => toml_ready(v) for (k, v) in d)
toml_ready(v::Bool) = v
toml_ready(v::Real) = isfinite(v) ? v : string(v)
toml_ready(v) = v
