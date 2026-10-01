# Deterministic energy-optimal re-optimization of the resampled reference
# (`reoptimize_reference_trajectory`: TrajectoryNLP without a seed). The Dopri8
# shooting defects are an ExaModels oracle with exact derivatives; everything else
# is native ExaModels algebra.

struct NominalParams
    n::Int
    ff_offset::Int     # local-vector offset of the feedforward block
    tmax::Float64
    mu::Float64
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
    rk_step(Dopri8, m, p.tmax * f, p.tmax * control_norm(f, p.eps), zero(SVector{7,eltype(m)}), h, p.mu, p.ve)

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
    _, stages = rk_step_stages(Dopri8, md, ud, p.tmax * nd, zero(SVector{7,PointDual}), h, p.mu, p.ve)
    xb, ub, mb = rk_adjoint(Dopri8, stages, ud, p.tmax * nd, g, h, p.mu, p.ve)
    zb = vcat(xb, p.tmax * ub + mb * (p.tmax * fd / nd))
    for r in 1:10, q in 1:r
        @inbounds vals[55(a-1)+lower_position(r, q)] = (partial(zb[r], q) + partial(zb[q], r)) / 2
    end
end

"Solve the energy-optimal problem on the CPU; returns the reoptimized reference and a report."
function reoptimize_reference(case::CaseData, o::Options, source::ReferenceTraj; output_file::String = "")
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
    p = NominalParams(n, NX * (n + 1), case.tmax, case.mu, case.ve, o.control_norm_eps)
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
