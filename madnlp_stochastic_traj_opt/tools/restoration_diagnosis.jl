# Diagnosis of a restoration phase. Replays a CPU solve with the default options up to
# the first failed line search at or after iteration AFTER, captures the iterate, the
# multipliers, the rejected step and the Hessian regularization, and then reports per
# constraint block (objective, mean defects, covariance defects, terminal covariance,
# thrust chance):
#   * the extreme eigenvalues of the block's contribution to the Lagrangian Hessian,
#     and its share of the most negative eigenvalue (v' H_b v for the eigenvector v);
#   * the curvature of each block along the rejected step;
#   * how far each block's linearization is off along the step, at the step lengths
#     of the backtracking line search.
#   julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/tools/restoration_diagnosis.jl [case] [after] [log]
# `log` is the MadNLP log of the run being replayed, for the reproduction check.
include(joinpath(@__DIR__, "..", "src", "CR3BPStochastic.jl"))
using .CR3BPStochastic
using LinearAlgebra, Printf, SparseArrays
import NLPModels, MadNLP
const C = CR3BPStochastic

const CASE = get(ARGS, 1, "lyapunov_l1_to_l2")
const AFTER = parse(Int, get(ARGS, 2, "60"))
const REFERENCE_LOG = get(ARGS, 3, "")

"MadNLP callback that stops at the first restoration request at or after iteration `after`."
mutable struct Capture <: MadNLP.AbstractUserCallback
    after::Int
    state::Any
    history::Vector{Any}      # per regular iteration: k, mu, del_w of the previous step, |y| of the unscaled rows
end
Capture(after) = Capture(after, nothing, Any[])
(c::Capture)(solver, ::MadNLP.AbstractUserCallbackStatus) = true
function (c::Capture)(solver, ::MadNLP.UserCallbackRegular)
    y = MadNLP.get_y(solver) .* MadNLP.get_cb(solver).con_scale
    push!(c.history, (k = MadNLP.get_cnt(solver).k, mu = MadNLP.get_mu(solver), del_w = MadNLP.get_del_w(solver),
        ymax = maximum(abs, y; init = 0.0), inf_pr = MadNLP.get_inf_pr(solver), y = y))
    return true
end
const CaptureSolver = MadNLP.MadNLPSolver{T,VT,VI,KKT,Model,CB,Iterator,IC,KV,Capture} where
    {T,VT,VI,KKT,Model,CB,Iterator,IC,KV}

function capture!(solver, kind)
    x, d = MadNLP.get_x(solver), MadNLP.get_d(solver)
    MadNLP.get_intermediate_callback(solver).state = (; kind, k = MadNLP.get_cnt(solver).k,
        x = copy(MadNLP.variable(x)), xfull = copy(MadNLP.full(x)), y = copy(MadNLP.get_y(solver)),
        dfull = copy(MadNLP.primal(d)), dy = copy(MadNLP.dual(d)), del_w = MadNLP.get_del_w(solver),
        mu = MadNLP.get_mu(solver), tau = MadNLP.get_tau(solver),
        xl = copy(MadNLP.primal(MadNLP.get_xl(solver))), xu = copy(MadNLP.primal(MadNLP.get_xu(solver))),
        zl = copy(MadNLP.primal(MadNLP.get_zl(solver))), zu = copy(MadNLP.primal(MadNLP.get_zu(solver))),
        obj_scale = MadNLP.get_cb(solver).obj_scale[], con_scale = copy(MadNLP.get_cb(solver).con_scale))
    return MadNLP.USER_REQUESTED_STOP
end
native_soft(solver) = invoke(MadNLP.restore!, Tuple{MadNLP.AbstractMadNLPSolver}, solver)
native_robust(solver) = invoke(MadNLP.robust!, Tuple{MadNLP.AbstractMadNLPSolver}, solver)
function MadNLP.restore!(solver::CaptureSolver)
    c = MadNLP.get_intermediate_callback(solver)
    MadNLP.get_cnt(solver).k >= c.after && c.state === nothing && return capture!(solver, :soft)
    return Base.invokelatest(native_soft, solver)::MadNLP.Status
end
function MadNLP.robust!(solver::CaptureSolver)
    c = MadNLP.get_intermediate_callback(solver)
    MadNLP.get_cnt(solver).k >= c.after && c.state === nothing && return capture!(solver, :robust)
    return Base.invokelatest(native_robust, solver)::MadNLP.Status
end

function dense_hessian(model, x, y, obj_weight)
    rows, cols = NLPModels.hess_structure(model)
    H = Matrix(sparse(rows, cols, NLPModels.hess_coord(model, x, y; obj_weight), model.meta.nvar, model.meta.nvar))
    return Symmetric(H + H' - Diagonal(diag(H)))
end

function diagnose(case_id, after)
    o = C.Options(; case_id)
    case = C.case_data(case_id); normalization = C.build_normalization(case); p = C.ut_params(case, o, normalization)
    scratch = mktempdir()
    ref, _ = C.reoptimize_reference(case, o, C.load_reference(case, o); output_file = joinpath(scratch, "nominal.log"))
    seed = C.build_initial_guess(case, o, normalization, p, ref)
    scaling = C.covariance_scaling(seed, normalization)
    blocks = C.build_stochastic_model(case, o, ref, seed, scaling, p, nothing)
    model, n = blocks.model, C.n_arcs(ref)
    log_file = joinpath(scratch, "stochastic.log")
    settings = merge(C.stochastic_solver_options(o; output_file = log_file), (; intermediate_callback = Capture(after)))
    stats, solver = C.run_madnlp(model, settings)
    s = MadNLP.get_intermediate_callback(solver).state
    s === nothing && error("no restoration request at or after iteration $after (status $(stats.status))")

    # ---- reproduction check against the original run's log
    iteration_line(file, k) = (m = match(Regex("^\\s*$(k)\\s+\\S+\\s+\\S+\\s+\\S+", "m"), read(file, String)); m === nothing ? "-" : strip(m.match))
    println("\n== reproduction: failed line search from iteration $(s.k) ($(s.kind) restoration requested)")
    for k in max(0, s.k - 3):s.k
        println("   replay   ", iteration_line(log_file, k))
        isempty(REFERENCE_LOG) || println("   original ", iteration_line(REFERENCE_LOG, k))
    end

    # ---- growth of the multipliers
    history = MadNLP.get_intermediate_callback(solver).history
    rows = [("mean defects", 14:13+7n), ("covariance defects", 14+7n:13+35n), ("terminal covariance", 14+35n:34+35n),
            ("thrust chance", 35+35n:34+36n)]
    varblocks = [(name, v.offset+1:v.offset+v.length) for (name, v) in
                 (("means", blocks.means), ("feedforward", blocks.ff), ("gains", blocks.gains), ("Cholesky factors", blocks.chol),
                  ("terminal margin", blocks.margin))]
    println("\n== multipliers of the unscaled rows by iteration (max |y| per block)")
    @printf("   %5s %9s %10s %10s %11s %11s %11s %11s\n", "k", "mu", "del_w", "inf_pr", "mean def.", "cov. def.", "terminal", "chance")
    for h in history
        (h.k <= 10 || h.k % 5 == 0 || h.k >= s.k - 3) || continue
        blockmax = [maximum(abs, h.y[r]; init = 0.0) for (_, r) in rows]
        @printf("   %5d %9.1e %10.1e %10.2e %11.2e %11.2e %11.2e %11.2e\n", h.k, h.mu, h.del_w, h.inf_pr, blockmax...)
    end
    x, nvar = s.x, model.meta.nvar
    y = s.y .* s.con_scale                          # multipliers of the unscaled constraints
    dx = s.dfull[1:nvar]
    @printf("\n== state: mu %.1e, Hessian regularization del_w %.2e, |dx| %.2e, |dy| %.2e\n", s.mu, s.del_w, norm(dx), norm(s.dy))
    println("   multipliers |y| (max) per block: ", join((@sprintf("%s %.2e", b, maximum(abs, y[r])) for (b, r) in rows), ", "))
    println("   step |dx| per variable block:     ", join((@sprintf("%s %.2e", b, norm(dx[r])) for (b, r) in varblocks), ", "))

    # ---- Hessian decomposition
    parts = [("objective", dense_hessian(model, x, zeros(model.meta.ncon), s.obj_scale))]
    for (name, r) in rows
        yb = zeros(model.meta.ncon); yb[r] = y[r]
        push!(parts, (name, dense_hessian(model, x, yb, 0.0)))
    end
    H = dense_hessian(model, x, y, s.obj_scale)
    @printf("\n== Lagrangian Hessian (%d x %d): assembled-from-blocks error %.1e\n", nvar, nvar,
        maximum(abs, H - sum(last.(parts))) / maximum(abs, H))
    E = eigen(H)
    v = E.vectors[:, 1]
    sigma = zeros(nvar)                                           # barrier diagonal of the bounded variables
    for i in 1:nvar
        isfinite(s.xl[i]) && (sigma[i] += s.zl[i] / (s.xfull[i] - s.xl[i]))
        isfinite(s.xu[i]) && (sigma[i] += s.zu[i] / (s.xu[i] - s.xfull[i]))
    end
    @printf("   eigenvalues: min %.3e, max %.3e, %d negative; with the barrier diagonal: min %.3e\n",
        E.values[1], E.values[end], count(<(0), E.values), eigmin(Symmetric(Matrix(H) + Diagonal(sigma))))
    @printf("   %-22s %12s %12s %16s %18s\n", "block", "min eig", "max eig", "share of min eig", "curvature along dx")
    for (name, Hb) in parts
        ev = eigvals(Hb)
        @printf("   %-22s %12.3e %12.3e %16.3e %18.3e\n", name, ev[1], ev[end], dot(v, Hb * v), dot(dx, Hb * dx) / dot(dx, dx))
    end
    @printf("   %-22s %12s %12s %16.3e %18.3e\n", "total", "", "", E.values[1], dot(dx, H * dx) / dot(dx, dx))
    println("   eigenvector of the min eigenvalue, energy per variable block: ",
        join((@sprintf("%s %.2f", b, sum(abs2, v[r])) for (b, r) in varblocks), ", "))
    arc_energy = [sum(abs2, v[r]) for r in [blocks.gains.offset .+ (18(a-1)+1:18a) for a in 1:n]] .+
                 [a == 1 ? 0.0 : sum(abs2, v[blocks.chol.offset .+ (28(a-2)+1:28(a-1))]) for a in 1:n]
    top = sortperm(arc_energy; rev = true)[1:5]
    println("   ... arcs carrying most of it (gains + incoming factor): ", join((@sprintf("%d (%.2f)", a, arc_energy[a]) for a in top), ", "))

    mean_arc = [sum(abs2, v[blocks.means.offset .+ (7(a-1)+1:7a)]) for a in 1:n+1]
    mean_comp = [sum(abs2, v[blocks.means.offset .+ (i:7:7(n+1))]) for i in 1:7]
    println("   ... node means carrying it: ", join((@sprintf("node %d (%.2f)", a, mean_arc[a]) for a in sortperm(mean_arc; rev = true)[1:4]), ", "),
        "; by component (x y z vx vy vz m): ", join((@sprintf("%.2f", e) for e in mean_comp), " "))

    # ---- constraint qualification: near-dependent rows and the multipliers
    rowsJ, colsJ = NLPModels.jac_structure(model)
    J = Matrix(sparse(rowsJ, colsJ, NLPModels.jac_coord(model, x), model.meta.ncon, nvar))
    F = svd(J)
    sv = F.S
    u = F.U[:, end]                                            # left singular vector of the smallest singular value
    names = vcat([("initial state", 1:7), ("terminal mean", 8:13)], rows)
    @printf("\n== constraint Jacobian (%d x %d): largest singular value %.3e, smallest five %s\n", size(J)..., sv[1],
        join((@sprintf("%.2e", x) for x in sv[end-4:end]), " "))
    println("   rows of the most dependent combination (energy per block): ",
        join((@sprintf("%s %.2f", b, sum(abs2, u[r])) for (b, r) in names), ", "))
    yhat = y / norm(y)
    proj = [abs(dot(yhat, F.U[:, end-j])) for j in 0:9]
    @printf("   multipliers along the 10 smallest left singular vectors: |cos| %s (sum of squares %.3f)\n",
        join((@sprintf("%.2f", p) for p in proj), " "), sum(abs2, proj))
    @printf("   |J' y| = %.3e while |y| = %.3e (cancellation ratio %.1e)\n", norm(J' * y), norm(y), norm(J' * y) / (opnorm(J) * norm(y)))
    top_rows = sortperm(abs.(u); rev = true)[1:8]
    row_label(i) = begin
        for (b, r) in names
            i in r || continue
            b == "covariance defects" && (q = i - first(r); return @sprintf("%s arc %d entry (%d,%d)", b, q ÷ 28 + 1, C.LTRI_ROWS[q % 28 + 1], C.LTRI_COLS[q % 28 + 1]))
            b == "mean defects" && (q = i - first(r); return @sprintf("%s arc %d comp %d", b, q ÷ 7 + 1, q % 7 + 1))
            return @sprintf("%s row %d", b, i - first(r) + 1)
        end
        return "row $i"
    end
    println("   largest rows of that combination: ", join((@sprintf("%s (%.2f)", row_label(i), u[i]) for i in top_rows), "; "))

    # ---- which bound limits the step (fraction to the boundary)
    ratios = fill(Inf, length(s.xfull))
    for i in eachindex(s.xfull)
        s.dfull[i] < 0 && isfinite(s.xl[i]) && (ratios[i] = -s.tau * (s.xfull[i] - s.xl[i]) / s.dfull[i])
        s.dfull[i] > 0 && isfinite(s.xu[i]) && (ratios[i] = s.tau * (s.xu[i] - s.xfull[i]) / s.dfull[i])
    end
    limiting = sortperm(ratios)[1:3]
    what(i) = i <= nvar ? join((b for (b, r) in varblocks if i in r)) * " variable $i" : "slack $(i - nvar) (thrust chance row)"
    println("   step limited by: ", join((@sprintf("%s (alpha %.2e, distance %.2e, d %.2e)", what(i), ratios[i],
        s.xfull[i] - (s.dfull[i] < 0 ? s.xl[i] : s.xu[i]), s.dfull[i]) for i in limiting), "; "))

    # ---- linearization along the rejected step
    alpha_max = MadNLP.get_alpha_max(s.xfull, s.xl, s.xu, s.dfull, s.tau)
    c0 = NLPModels.cons(model, x)
    Jdx = NLPModels.jprod(model, x, dx)
    target = model.meta.lcon
    @printf("\n== linearization along the rejected step (alpha_max %.3e)\n", alpha_max)
    @printf("   %-12s %-22s %14s %14s %14s %14s\n", "alpha", "block", "violation now", "linear model", "actual", "nonlinear/lin")
    for j in (0, 4, 8, 12, 16, 20)
        alpha = alpha_max * 2.0^-j
        c1 = NLPModels.cons(model, x + alpha * dx)
        for (name, r) in rows[1:3]                               # the equality blocks
            now = maximum(abs, c0[r] - target[r]); lin = maximum(abs, c0[r] + alpha * Jdx[r] - target[r])
            act = maximum(abs, c1[r] - target[r])
            ratio = maximum(abs, c1[r] - c0[r] - alpha * Jdx[r]) / max(maximum(abs, alpha * Jdx[r]), 1e-300)
            @printf("   %-12.3e %-22s %14.3e %14.3e %14.3e %14.3e\n", alpha, name, now, lin, act, ratio)
        end
        r = rows[4][2]
        ratio = maximum(abs, c1[r] - c0[r] - alpha * Jdx[r]) / max(maximum(abs, alpha * Jdx[r]), 1e-300)
        @printf("   %-12.3e %-22s %14.3e %14s %14.3e %14.3e\n", alpha, "thrust chance (max c)", maximum(c0[r]), "", maximum(c1[r]), ratio)
    end
end

Base.invokelatest(diagnose, CASE, AFTER)
