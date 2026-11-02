# Deterministic energy-optimal re-optimization of the resampled reference
# (`reoptimize_reference_trajectory`: TrajectoryNLP without a seed). The Dopri8
# shooting defects are an ExaModels oracle with exact derivatives; everything else
# is native ExaModels algebra.

struct NominalParams{G<:GravityModel}
    n::Int
    ff_offset::Int     # local-vector offset of the feedforward block
    tmax::Float64
    gravity::G
    ve::Float64
    eps::Float64
end

@inline function nominal_inputs(x, a, p::NominalParams)
    m = SVector{7}(ntuple(i -> @inbounds(x[7(a-1)+i]), Val(7)))
    f = SVector{3}(ntuple(i -> @inbounds(x[p.ff_offset+3(a-1)+i]), Val(3)))
    return m, f
end

"Dopri8 arc map with thrust T_max f and regularized mass-flow magnitude T_max ||f||_eps."
@inline nominal_map(m, f, h, p::NominalParams) =
    rk_step(Dopri8, m, p.tmax * f, p.tmax * control_norm(f, p.eps), zero(SVector{7,eltype(m)}), h, p.gravity, p.ve)

@kernel function nominal_value_kernel!(out, @Const(x), @Const(steps), p::NominalParams)
    a = @index(Global, Linear)
    m, f = nominal_inputs(x, a, p)
    y = nominal_map(m, f, @inbounds(steps[a]), p)
    for i in 1:7
        @inbounds out[7(a-1)+i] = -y[i]
    end
end

@kernel function nominal_jacobian_kernel!(vals, @Const(x), @Const(steps), p::NominalParams)
    a = @index(Global, Linear)
    m, f = nominal_inputs(x, a, p)
    y = nominal_map(seed(TagPoint, m, 0, Val(10)), seed(TagPoint, f, 7, Val(10)), @inbounds(steps[a]), p)
    for i in 1:7, q in 1:10
        @inbounds vals[70(a-1)+10(i-1)+q] = -partial(y[i], q)
    end
end

@kernel function nominal_hessian_kernel!(vals, @Const(x), @Const(multipliers), @Const(steps), p::NominalParams)
    a = @index(Global, Linear)
    m, f = nominal_inputs(x, a, p)
    g = SVector{7}(ntuple(i -> -@inbounds(multipliers[7(a-1)+i]), Val(7)))
    md, fd = seed(TagPoint, m, 0, Val(10)), seed(TagPoint, f, 7, Val(10))
    ud, nd = p.tmax * fd, control_norm(fd, p.eps)
    h = @inbounds steps[a]
    _, stages = rk_step_stages(Dopri8, md, ud, p.tmax * nd, zero(SVector{7,PointDual}), h, p.gravity, p.ve)
    xb, ub, mb = rk_adjoint(Dopri8, stages, ud, p.tmax * nd, g, h, p.gravity, p.ve)
    zb = vcat(xb, p.tmax * ub + mb * (p.tmax * fd / nd))
    for r in 1:10, q in 1:r
        @inbounds vals[55(a-1)+lower_position(r, q)] = (partial(zb[r], q) + partial(zb[q], r)) / 2
    end
end

"""
Dopri8 arc map of the deterministic optimisation's controls from theta = (start state 7, direction 3, throttle):
thrust T_max Gamma d and mass flow T_max Gamma (no smoothing).
"""
@inline throttle_nominal_map(case::CaseData, theta, h) = rk_step(Dopri8, SVector{7}(ntuple(i -> theta[i], Val(7))),
    case.tmax * theta[11] * SVector{3}(theta[8], theta[9], theta[10]), case.tmax * theta[11],
    zero(SVector{7,Float64}), h, case.gravity, case.ve)

"""
Reoptimisation of the resampled reference in the deterministic optimisation's formulation: throttle
Gamma_k in [0, 1] and direction d_k with rows ||d_k||^2 = 1, objective 1 - m_f (fuel, the constant left out) or
the energy sum_k dt_k Gamma_k^2 ||d_k||^2 (`reference_objective`). Shooting defects by ForwardDiff on the CPU.
"""
function reoptimize_reference_throttle(case::CaseData, o::Options, source::ReferenceTraj; output_file::String = "")
    n = n_arcs(source)
    c = ExaCore(Float64)
    S0 = source.controls ./ case.tmax
    gamma0 = [norm(S0[:, k]) for k in 1:n]
    @add_var(c, means, 1:NX, 1:n+1; start = source.states)
    @add_var(c, ff, 1:NU, 1:n; start = reduce(hcat, [gamma0[k] > 0 ? S0[:, k] / gamma0[k] : [1.0, 0.0, 0.0] for k in 1:n]))
    @add_var(c, throttle, 1:n; start = min.(gamma0, 1.0), lvar = zeros(n), uvar = ones(n))
    if o.reference_objective == :fuel
        @add_obj(c, -means[7, t.node] for t in [(node = n + 1,)])
    else
        @add_obj(c, t.dt * throttle[t.k]^2 * (ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2)
            for t in [(k = k, dt = source.steps[k]) for k in 1:n])
    end
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
        "objective" => stats.objective, "objective_kind" => String(o.reference_objective),
        "max_constraint_violation" => original_violation(model, stats.solution),
        "restoration_iterations" => guard.restoration_iterations,
        "longest_restoration_run" => guard.max_consecutive)
    stats.status == MadNLP.SOLVE_SUCCEEDED || error("the $(o.reference_objective) reoptimisation did not converge: $(stats.status)")
    X = Array(ExaModels.solution(stats, means))
    U = Array(ExaModels.solution(stats, ff)) .* Array(ExaModels.solution(stats, throttle))' .* case.tmax
    return ReferenceTraj(copy(source.node_times), copy(source.steps), X, U, case.m0_wet * (X[7, 1] - X[7, end])), report
end

"Solve the energy-optimal problem on the CPU; returns the reoptimized reference and a report."
function reoptimize_reference(case::CaseData, o::Options, source::ReferenceTraj; output_file::String = "")
    o.control_parameterization == :throttle && return reoptimize_reference_throttle(case, o, source; output_file)
    n = n_arcs(source)
    c = ExaCore(Float64)
    @add_var(c, means, 1:NX, 1:n+1; start = source.states)
    @add_var(c, ff, 1:NU, 1:n; start = source.controls ./ case.tmax)
    epsilon2 = o.control_norm_eps^2
    @add_obj(c, t.dt * (ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2 + epsilon2)
        for t in [(k = k, dt = source.steps[k]) for k in 1:n])
    @add_con(c, means[t.i, 1] - t.v for t in [(i = i, v = case.x0[i]) for i in 1:NX])
    target = terminal_mean_target(case, o, source)
    last = n + 1
    @add_con(c, means[t.i, last] - t.v for t in [(i = i, v = target[i]) for i in 1:NP])
    @add_con(c, defects, means[t.i, t.k+1] for t in [(i = i, k = k) for k in 1:n for i in 1:NX])
    @add_con(c, sqrt(ff[1, k]^2 + ff[2, k]^2 + ff[3, k]^2 + epsilon2) for k in 1:n; lcon = -Inf, ucon = 1.0)
    p = NominalParams(n, NX * (n + 1), case.tmax, case.gravity, case.ve, o.control_norm_eps)
    steps = copy(source.steps)
    local_index(a, q) = q <= 7 ? 7(a-1) + q : p.ff_offset + 3(a-1) + (q - 7)
    jac_rows = [7(a-1)+i for a in 1:n for i in 1:7 for _ in 1:10]
    jac_cols = [local_index(a, q) for a in 1:n for _ in 1:7 for q in 1:10]
    hess_rows = [local_index(a, r) for a in 1:n for r in 1:10 for _ in 1:r]
    hess_cols = [local_index(a, q) for a in 1:n for r in 1:10 for q in 1:r]
    c, _ = ExaModels.add_eval(c, (defects,), (means, ff),
        (out, x) -> launch!(nominal_value_kernel!, n, out, x, steps, p);
        jac! = (vals, x) -> launch!(nominal_jacobian_kernel!, n, vals, x, steps, p),
        hess! = (vals, x, y) -> launch!(nominal_hessian_kernel!, n, vals, x, y, steps, p),
        jac_structure! = (r, cc) -> (append!(r, jac_rows); append!(cc, jac_cols)),
        hess_structure! = (r, cc) -> (append!(r, hess_rows); append!(cc, hess_cols)))
    model = ExaModel(c)
    model.meta.nvar == NX * (n + 1) + NU * n || error("unexpected energy-optimal variable count")
    settings = nominal_solver_options(o; output_file)
    guard = settings.intermediate_callback
    stats, _ = run_madnlp(model, settings)
    report = Dict{String,Any}("status" => string(stats.status), "iterations" => stats.iter,
        "objective" => stats.objective, "max_constraint_violation" => original_violation(model, stats.solution),
        "restoration_iterations" => guard.restoration_iterations,
        "longest_restoration_run" => guard.max_consecutive)
    stats.status == MadNLP.SOLVE_SUCCEEDED || error("the energy-optimal reference did not converge: $(stats.status)")
    X = Array(ExaModels.solution(stats, means))
    U = Array(ExaModels.solution(stats, ff)) .* case.tmax
    ref = ReferenceTraj(copy(source.node_times), copy(source.steps), X, U, case.m0_wet * (X[7, 1] - X[7, end]))
    return ref, report
end
