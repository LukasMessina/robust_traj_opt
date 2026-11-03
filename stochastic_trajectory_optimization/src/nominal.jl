# Fuel-optimal re-optimization of the resampled deterministic reference on the stochastic mesh, in the
# deterministic optimisation's formulation (throttle and unit direction). The Dopri8 shooting defects are an
# ExaModels oracle with exact ForwardDiff derivatives; everything else is native ExaModels algebra.

"""
Dopri8 arc map of the deterministic optimisation's controls from theta = (start state 7, direction 3, throttle):
thrust T_max Gamma d and mass flow T_max Gamma (no smoothing).
"""
@inline throttle_nominal_map(case::CaseData, theta, h) = rk_step(Dopri8, SVector{7}(ntuple(i -> theta[i], Val(7))),
    case.tmax * theta[11] * SVector{3}(theta[8], theta[9], theta[10]), case.tmax * theta[11],
    zero(SVector{7,Float64}), h, case.gravity, case.ve)

"""
Reoptimisation of the resampled reference on the CPU: throttle Gamma_k in [0, 1] and direction d_k with rows
||d_k||^2 = 1, objective 1 - m_f (the constant left out). Returns the reoptimised reference and a report.
"""
function reoptimize_reference(case::CaseData, o::Options, source::ReferenceTraj; output_file::String = "")
    n = n_arcs(source)
    c = ExaCore(Float64)
    S0 = source.controls ./ case.tmax
    gamma0 = [norm(S0[:, k]) for k in 1:n]
    @add_var(c, means, 1:NX, 1:n+1; start = source.states)
    @add_var(c, ff, 1:NU, 1:n; start = reduce(hcat, [gamma0[k] > 0 ? S0[:, k] / gamma0[k] : [1.0, 0.0, 0.0] for k in 1:n]))
    @add_var(c, throttle, 1:n; start = min.(gamma0, 1.0), lvar = zeros(n), uvar = ones(n))
    @add_obj(c, -means[7, t.node] for t in [(node = n + 1,)])
    @add_con(c, means[t.i, 1] - t.v for t in [(i = i, v = case.x0[i]) for i in 1:NX])
    target = terminal_mean_target(case, o, source)
    @add_con(c, means[t.i, t.node] - t.v for t in [(i = i, node = n + 1, v = target[i]) for i in 1:NP])
    @add_con(c, defects, means[t.i, t.k+1] for t in [(i = i, k = k) for k in 1:n for i in 1:NX])
    @add_con(c, ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2 - 1 for t in [(k = k,) for k in 1:n])
    index = [vcat(means.offset .+ 7(k-1) .+ (1:7), ff.offset .+ 3(k-1) .+ (1:3), throttle.offset + k) for k in 1:n]
    load(x, k) = SVector{11}(ntuple(q -> x[index[k][q]], Val(11)))
    steps = copy(source.steps)
    jac_rows = [7(k-1) + i for k in 1:n for i in 1:NX for _ in 1:11]
    jac_cols = [index[k][q] for k in 1:n for _ in 1:NX for q in 1:11]
    hess_rows, hess_cols = Int[], Int[]
    for k in 1:n
        lower_structure!(hess_rows, hess_cols, index[k])
    end
    c, _ = ExaModels.add_eval(c, (defects,), (means, ff, throttle),
        function (out, x)
            Threads.@threads for k in 1:n
                out[7(k-1) .+ (1:7)] = -throttle_nominal_map(case, load(x, k), steps[k])
            end
        end;
        jac! = function (vals, x)
            Threads.@threads for k in 1:n
                J = ForwardDiff.jacobian(v -> -throttle_nominal_map(case, v, steps[k]), load(x, k))
                vals[77(k-1) .+ (1:77)] = vec(permutedims(J))
            end
        end,
        hess! = function (vals, x, y)
            Threads.@threads for k in 1:n
                g = SVector{7}(ntuple(i -> -y[7(k-1)+i], Val(7)))
                H = ForwardDiff.hessian(v -> dot(g, throttle_nominal_map(case, v, steps[k])), load(x, k))
                lower_entries!(vals, 66(k-1), H, index[k])
            end
        end,
        jac_structure! = (r, cc) -> (append!(r, jac_rows); append!(cc, jac_cols)),
        hess_structure! = (r, cc) -> (append!(r, hess_rows); append!(cc, hess_cols)))
    model = ExaModel(c)
    model.meta.nvar == NX * (n + 1) + (NU + 1) * n || error("unexpected throttle reoptimisation variable count")
    settings = nominal_solver_options(o; output_file)
    guard = settings.intermediate_callback
    stats, _ = run_madnlp(model, settings)
    report = Dict{String,Any}("status" => string(stats.status), "iterations" => stats.iter,
        "objective" => stats.objective, "objective_kind" => "fuel",
        "max_constraint_violation" => original_violation(model, stats.solution),
        "restoration_iterations" => guard.restoration_iterations,
        "longest_restoration_run" => guard.max_consecutive)
    stats.status == MadNLP.SOLVE_SUCCEEDED || error("the fuel reoptimisation did not converge: $(stats.status)")
    X = Array(ExaModels.solution(stats, means))
    U = Array(ExaModels.solution(stats, ff)) .* Array(ExaModels.solution(stats, throttle))' .* case.tmax
    return ReferenceTraj(copy(source.node_times), copy(source.steps), X, U, case.m0_wet * (X[7, 1] - X[7, end])), report
end
