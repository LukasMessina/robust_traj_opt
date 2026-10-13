# Stochastic multiple-shooting NLP (the seeded TrajectoryNLP of the JAX script).
#
# Variables (56 N + 28, the reference count): node means 7(N+1), normalized
# feedforward 3N, scaled gains K_hat = K_tilde / gain_scale 18N, seed-scaled node
# Cholesky factors of nodes 1..N 28N, terminal margin factor 21.
# Constraints (36 N + 34): initial state 7, terminal mean 6, mean defects 7N,
# covariance defects 28N, terminal residual 21, thrust chance N.
# Bounds: nonnegative factor diagonals, margin diagonal >= terminal_margin_floor,
# scaled gains |K_hat| <= gain_bound.
# chance_formulation = :slack adds tau_hat (N) and the packed 3x3 factor G_hat (6N), the
# chance row becomes ||S_k||_eps + psi s_k tau_hat_k <= 1, and 6N rows
# tau_hat_k^2 I - T_k / s_k^2 - G_hat_k G_hat_k' = 0 follow it (s_k = sqrt(lambda_max(T_k)) at
# the seed); 63 N + 28 variables and 42 N + 34 constraints.
#
# Objective per arc k (unweighted, equal arcs; w = control_covariance_weight):
#   ||S_k||_eps + tr(P_k[1:6,1:6]) / c^2 + w tr(T_k),
# with tr(T_k) in the closed form of `control_covariance`:
#   tr(T) = (1 + s2 + 2 s4) ct tr(K M K') + (s2 + 2 s4) ||f||^2 + 3 eps,
#   tr(K M K') = d tr(K P6 K') + sum_qb K_qb^2 (eps + d R_nav,b).
# Terms constant in the variables (tr(P_0)/c^2 and 3 eps per arc) are left out
# of the NLP and returned as `objective_constant`.

"Row (a, b) of pack(L L') as `terms` products, padded with zero-weight products."
function packed_product_terms(a, b, terms)
    pairs = [(packed_index(a, c), packed_index(b, c), 1.0) for c in 1:b]
    while length(pairs) < terms
        push!(pairs, (packed_index(b, b), packed_index(b, b), 0.0))
    end
    return pairs
end

"Named tuple fields a1, b1, m1, ... of the padded products of row (i, j)."
function product_fields(i, j, terms)
    t = packed_product_terms(i, j, terms)
    names = Tuple(Symbol(prefix, k) for k in 1:terms for prefix in (:a, :b, :m))
    values = Tuple(v for k in 1:terms for v in t[k])
    return NamedTuple{names}(values)
end

"Control covariance T_k of every arc at the model's start point, as `control_covariance` evaluates it."
function start_control_covariances(seed::InitialGuess, scaling::CovarianceScaling, p::UTParams, n::Int)
    return map(1:n) do a
        L = a == 1 ? scaling.initial_factor : scaling.factors[:, a-1]
        P6 = position_velocity_covariance(SVector{7}(scaling.std[:, a]), SVector{21}(L[1:21]))
        Matrix(control_covariance(p, SVector{3}(seed.feedforward[:, a]), SMatrix{3,6}(seed.gains[:, :, a]), P6))
    end
end

"""
Scale s = sqrt(lambda_max(T)) and the scaled start of the slack chance variables: tau^2 =
lambda_max + 4 floor^2 (the smallest G pivot starts near 2 floor) and G = chol(tau^2 I - T).
lambda_max comes from the LAPACK symmetric eigensolver, not the smoothed formula.
"""
function slack_chance_start(T::AbstractMatrix, floor)
    lambda_max = eigmax(Symmetric(T))
    s = sqrt(max(lambda_max, 1e-16))
    tau = sqrt(lambda_max + 4floor^2)
    G = cholesky(Symmetric(tau^2 * I - T)).L
    return s, tau / s, pack_lower(Matrix(G)) ./ s
end

"""
The ExaModels model and its variable blocks. `ut` holds the UT oracle's index data
and workspace (used by the tools that call the oracle directly).
"""
function build_stochastic_model(case::CaseData, o::Options, ref::ReferenceTraj, seed::InitialGuess,
        scaling::CovarianceScaling, p::UTParams, backend)
    n = n_arcs(ref)
    c = ExaCore(Float64; backend)
    lvar_chol = fill(-Inf, NL, n)
    lvar_chol[collect(LTRI_DIAGONAL), :] .= 0.0
    lvar_margin = fill(-Inf, NM)
    lvar_margin[collect(MTRI_DIAGONAL)] .= o.terminal_margin_floor
    @add_var(c, means, 1:NX, 1:n+1; start = seed.means)
    @add_var(c, ff, 1:NU, 1:n; start = seed.feedforward)
    @add_var(c, gains, 1:NU, 1:NP, 1:n; start = seed.gains ./ reshape(seed.gain_scale, 1, 1, :),
        lvar = fill(-o.gain_bound, NU, NP, n), uvar = fill(o.gain_bound, NU, NP, n))
    @add_var(c, chol, 1:NL, 1:n; start = scaling.factors, lvar = lvar_chol)
    @add_var(c, margin, 1:NM; start = seed.terminal_margin, lvar = lvar_margin)
    slack = o.chance_formulation == :slack
    if slack
        starts = [slack_chance_start(T, o.chance_slack_floor) for T in start_control_covariances(seed, scaling, p, n)]
        slack_scale = [first(s) for s in starts]
        lvar_slack = fill(-Inf, NU * (NU + 1) ÷ 2, n)
        for a in 1:n, r in 1:NU
            lvar_slack[packed_index(r, r), a] = o.chance_slack_floor / slack_scale[a]
        end
        @add_var(c, tau, 1:n; start = [s[2] for s in starts], lvar = zeros(n))
        @add_var(c, slack_factor, 1:NU*(NU+1)÷2, 1:n; start = reduce(hcat, [s[3] for s in starts]), lvar = lvar_slack)
    end

    # ------------------------------------------------------------ objective
    # A feasibility solve has no objective terms (and no objective constant).
    if !o.feasibility_only
        epsilon2 = o.control_norm_eps^2
        @add_obj(c, sqrt(ff[1, k]^2 + ff[2, k]^2 + ff[3, k]^2 + epsilon2) for k in 1:n)
        trace_data = [(k = a - 1, r = r, w = scaling.std[LTRI_ROWS[r], a]^2 * p.inv_sigma_sq) for a in 2:n for r in 1:NM]
        @add_obj(c, t.w * chol[t.r, t.k]^2 for t in trace_data)
        cA = o.control_covariance_weight * (1 + p.var_mag + 2p.var_point)
        cf = o.control_covariance_weight * (p.var_mag + 2p.var_point)
        # tr(K P6 K') = sum_{q, col} (sum_{row >= col} K_{q,row} std_row L_{row,col})^2
        L0 = unpack_lower(scaling.initial_factor, NX)
        kp_data = [begin
            weights = [row >= col ? scaling.std[row, a] : 0.0 for row in 1:NP]
            index(row) = packed_index(max(row, col), col)
            (a = a, k = a - 1, q = q, w = cA * p.ct * p.d * seed.gain_scale[a]^2,
             c1 = weights[1], r1 = index(1), c2 = weights[2], r2 = index(2), c3 = weights[3], r3 = index(3),
             c4 = weights[4], r4 = index(4), c5 = weights[5], r5 = index(5), c6 = weights[6], r6 = index(6))
        end for a in 2:n for q in 1:NU for col in 1:NP]
        kp0_data = [begin
            weights = [row >= col ? scaling.std[row, 1] * L0[row, col] : 0.0 for row in 1:NP]
            (q = q, w = cA * p.ct * p.d * seed.gain_scale[1]^2, c1 = weights[1], c2 = weights[2], c3 = weights[3],
             c4 = weights[4], c5 = weights[5], c6 = weights[6])
        end for q in 1:NU for col in 1:NP]
        @add_obj(c, t.w * (t.c1 * gains[t.q, 1, t.a] * chol[t.r1, t.k] + t.c2 * gains[t.q, 2, t.a] * chol[t.r2, t.k] +
                           t.c3 * gains[t.q, 3, t.a] * chol[t.r3, t.k] + t.c4 * gains[t.q, 4, t.a] * chol[t.r4, t.k] +
                           t.c5 * gains[t.q, 5, t.a] * chol[t.r5, t.k] + t.c6 * gains[t.q, 6, t.a] * chol[t.r6, t.k])^2
            for t in kp_data)
        @add_obj(c, t.w * (t.c1 * gains[t.q, 1, 1] + t.c2 * gains[t.q, 2, 1] + t.c3 * gains[t.q, 3, 1] +
                           t.c4 * gains[t.q, 4, 1] + t.c5 * gains[t.q, 5, 1] + t.c6 * gains[t.q, 6, 1])^2
            for t in kp0_data)
        nav_data = [(a = a, q = q, b = b, w = cA * p.ct * seed.gain_scale[a]^2 * (p.jitter + p.d * p.nav_var[b]))
                    for a in 1:n for b in 1:NP for q in 1:NU]
        @add_obj(c, t.w * gains[t.q, t.b, t.a]^2 for t in nav_data)
        @add_obj(c, cf * (ff[1, k]^2 + ff[2, k]^2 + ff[3, k]^2) for k in 1:n)
    end

    # ---------------------------------------------------------- constraints
    @add_con(c, means[t.i, 1] - t.v for t in [(i = i, v = case.x0[i]) for i in 1:NX])
    target = terminal_mean_target(case, o, ref)
    last = n + 1
    @add_con(c, means[t.i, last] - t.v for t in [(i = i, v = target[i]) for i in 1:NP])
    # Native parts of the defect rows; the UT oracle below adds the predicted moments.
    @add_con(c, mean_defects, means[t.i, t.k+1] for t in [(i = i, k = a) for a in 1:n for i in 1:NX])
    cov_data = [merge((k = a,), product_fields(LTRI_ROWS[r], LTRI_COLS[r], NX)) for a in 1:n for r in 1:NL]
    @add_con(c, cov_defects, t.m1 * chol[t.a1, t.k] * chol[t.b1, t.k] + t.m2 * chol[t.a2, t.k] * chol[t.b2, t.k] +
        t.m3 * chol[t.a3, t.k] * chol[t.b3, t.k] + t.m4 * chol[t.a4, t.k] * chol[t.b4, t.k] +
        t.m5 * chol[t.a5, t.k] * chol[t.b5, t.k] + t.m6 * chol[t.a6, t.k] * chol[t.b6, t.k] +
        t.m7 * chol[t.a7, t.k] * chol[t.b7, t.k] for t in cov_data)
    # Terminal covariance bound P'_N <= Dt (matrix inequality):  I - Dt^-1/2 P'_N Dt^-1/2 - G G' = 0.
    inverse_target = terminal_inverse_std(o)
    std_N = scaling.std[:, n+1]
    terminal_data = [begin
        i, j = LTRI_ROWS[r], LTRI_COLS[r]
        merge((delta = Float64(i == j), w = std_N[i] * std_N[j] * inverse_target[i] * inverse_target[j]),
            product_fields(i, j, NP))
    end for r in 1:NM]
    @add_con(c, t.delta -
        t.w * (t.m1 * chol[t.a1, n] * chol[t.b1, n] + t.m2 * chol[t.a2, n] * chol[t.b2, n] +
               t.m3 * chol[t.a3, n] * chol[t.b3, n] + t.m4 * chol[t.a4, n] * chol[t.b4, n] +
               t.m5 * chol[t.a5, n] * chol[t.b5, n] + t.m6 * chol[t.a6, n] * chol[t.b6, n]) -
        (t.m1 * margin[t.a1] * margin[t.b1] + t.m2 * margin[t.a2] * margin[t.b2] +
         t.m3 * margin[t.a3] * margin[t.b3] + t.m4 * margin[t.a4] * margin[t.b4] +
         t.m5 * margin[t.a5] * margin[t.b5] + t.m6 * margin[t.a6] * margin[t.b6]) for t in terminal_data)

    # --------------------------------------------------------------- oracles
    offsets = (; means = means.offset, ff = ff.offset, gains = gains.offset, chol = chol.offset)
    offsets.means == 0 && offsets.ff == NX * (n + 1) || error("unexpected variable layout")
    ut = ut_oracle(n, offsets, scaling, seed, ref.steps, p, backend)
    c, _ = ExaModels.add_eval(c, (mean_defects, cov_defects), (means, ff, gains, chol),
        (out, x) -> ut_values!(out, x, ut, p, n);
        jac! = (vals, x) -> ut_jacobian!(vals, x, ut, p, n),
        hess! = (vals, x, y) -> ut_hessian!(vals, x, y, ut, p, n),
        jac_structure! = (r, cc) -> (append!(r, ut.structure.jac_rows); append!(cc, ut.structure.jac_cols)),
        hess_structure! = (r, cc) -> (append!(r, ut.structure.hess_rows); append!(cc, ut.structure.hess_cols)))
    if slack
        c = add_slack_chance(c, n, scaling, seed, p, ff, gains, chol, tau, slack_factor, slack_scale)
    else
        c = ExaModels.constraint(c, chance_oracle(n, offsets, 56n + 28, scaling, seed, p, backend))
    end
    model = ExaModel(c)
    nvar, ncon = 56n + 28 + (slack ? 7n : 0), 36n + 34 + (slack ? 6n : 0)
    model.meta.nvar == nvar || error("variable count $(model.meta.nvar) differs from the expected $nvar")
    model.meta.ncon == ncon || error("constraint count $(model.meta.ncon) differs from the expected $ncon")
    P0 = node_covariance(scaling.initial_factor, scaling.std[:, 1])
    constant = o.feasibility_only ? 0.0 : tr(P0[1:NP, 1:NP]) * p.inv_sigma_sq + o.control_covariance_weight * n * 3p.jitter
    chance_slack = slack ? (; tau, factor = slack_factor, scale = slack_scale) : nothing
    return (; model, means, ff, gains, chol, margin, chance_slack, objective_constant = constant, ut)
end

"""
Slack thrust chance constraint (`chance_formulation = :slack`): the rows
||S_k||_eps + psi s_k tau_hat_k <= 1, then for every arc the packed lower triangle of
tau_hat_k^2 I - T_k / s_k^2 - G_hat_k G_hat_k' = 0. T_k is `control_covariance` written out
as a polynomial of (f, K_hat, L_tilde):

    T = alpha [d (K F)(K F)' + K diag(m) K'] + beta tr(d (K F)(K F)' + K diag(m) K') I + Q_G(f) + eps I,

with K = gain_scale K_hat, F = diag(std) L (6x6 block of the arc-start factor; the data L0 on
arc 1), m_b = eps + d R_nav,b, alpha = ct (1 + s2 - s4), beta = ct s4.
"""
function add_slack_chance(c, n, scaling, seed, p, ff, gains, chol, tau, slack_factor, slack_scale)
    chance_data = [(k = a, w = p.psi * slack_scale[a], e2 = p.eps^2) for a in 1:n]
    @add_con(c, sqrt(ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2 + t.e2) + t.w * tau[t.k] for t in chance_data;
        lcon = -Inf, ucon = 1.0)

    rows = [(k = a, i = i, j = j) for a in 1:n for i in 1:NU for j in 1:i]
    row_of = Dict((r.k, r.i, r.j) => id for (id, r) in enumerate(rows))
    kappa = 1.0 ./ slack_scale .^ 2
    base = [merge((k = r.k, delta = Float64(r.i == r.j), c0 = (r.i == r.j) * p.jitter * kappa[r.k]), product_fields(r.i, r.j, NU))
            for r in rows]
    @add_con(c, slack_rows, t.delta * tau[t.k]^2 - t.c0 -
        (t.m1 * slack_factor[t.a1, t.k] * slack_factor[t.b1, t.k] + t.m2 * slack_factor[t.a2, t.k] * slack_factor[t.b2, t.k] +
         t.m3 * slack_factor[t.a3, t.k] * slack_factor[t.b3, t.k]) for t in base)

    alpha = p.ct * (1 + p.var_mag - p.var_point)
    beta = p.ct * p.var_point
    m = [p.jitter + p.d * p.nav_var[b] for b in 1:NP]
    L0 = unpack_lower(scaling.initial_factor, NX)
    # (K F)_{q,col} = sum_row c_row K_hat[q,row] L[r_row]: weights and packed factor rows of column col
    kf_weights(a, col) = [row >= col ? scaling.std[row, a] : 0.0 for row in 1:NP]
    kf_index(col) = [packed_index(max(row, col), col) for row in 1:NP]
    kf_fields(a, col) = (w = kf_weights(a, col); r = kf_index(col);
        (c1 = w[1], r1 = r[1], c2 = w[2], r2 = r[2], c3 = w[3], r3 = r[3], c4 = w[4], r4 = r[4], c5 = w[5], r5 = r[5], c6 = w[6], r6 = r[6]))
    kf0_fields(col) = (w = [row >= col ? scaling.std[row, 1] * L0[row, col] : 0.0 for row in 1:NP];
        (c1 = w[1], c2 = w[2], c3 = w[3], c4 = w[4], c5 = w[5], c6 = w[6]))
    gs2(a) = seed.gain_scale[a]^2

    # -alpha d (K F)_i . (K F)_j and, on the diagonal, -beta d sum_q ||(K F)_q||^2 (arcs 2..n: factor variables)
    pair = [merge((id = row_of[(a, i, j)], a = a, k = a - 1, i = i, j = j, w = -kappa[a] * alpha * p.d * gs2(a)), kf_fields(a, col))
            for a in 2:n for i in 1:NU for j in 1:i for col in 1:NP]
    trace = [merge((id = row_of[(a, i, i)], a = a, k = a - 1, q = q, w = -kappa[a] * beta * p.d * gs2(a)), kf_fields(a, col))
             for a in 2:n for i in 1:NU for q in 1:NU for col in 1:NP]
    if !isempty(pair)
        @add_con!(c, slack_rows, t.id => t.w *
            (t.c1 * gains[t.i, 1, t.a] * chol[t.r1, t.k] + t.c2 * gains[t.i, 2, t.a] * chol[t.r2, t.k] +
             t.c3 * gains[t.i, 3, t.a] * chol[t.r3, t.k] + t.c4 * gains[t.i, 4, t.a] * chol[t.r4, t.k] +
             t.c5 * gains[t.i, 5, t.a] * chol[t.r5, t.k] + t.c6 * gains[t.i, 6, t.a] * chol[t.r6, t.k]) *
            (t.c1 * gains[t.j, 1, t.a] * chol[t.r1, t.k] + t.c2 * gains[t.j, 2, t.a] * chol[t.r2, t.k] +
             t.c3 * gains[t.j, 3, t.a] * chol[t.r3, t.k] + t.c4 * gains[t.j, 4, t.a] * chol[t.r4, t.k] +
             t.c5 * gains[t.j, 5, t.a] * chol[t.r5, t.k] + t.c6 * gains[t.j, 6, t.a] * chol[t.r6, t.k]) for t in pair)
        @add_con!(c, slack_rows, t.id => t.w *
            (t.c1 * gains[t.q, 1, t.a] * chol[t.r1, t.k] + t.c2 * gains[t.q, 2, t.a] * chol[t.r2, t.k] +
             t.c3 * gains[t.q, 3, t.a] * chol[t.r3, t.k] + t.c4 * gains[t.q, 4, t.a] * chol[t.r4, t.k] +
             t.c5 * gains[t.q, 5, t.a] * chol[t.r5, t.k] + t.c6 * gains[t.q, 6, t.a] * chol[t.r6, t.k])^2 for t in trace)
    end
    # the same on arc 1, whose factor L0 is data
    pair0 = [merge((id = row_of[(1, i, j)], i = i, j = j, w = -kappa[1] * alpha * p.d * gs2(1)), kf0_fields(col))
             for i in 1:NU for j in 1:i for col in 1:NP]
    trace0 = [merge((id = row_of[(1, i, i)], q = q, w = -kappa[1] * beta * p.d * gs2(1)), kf0_fields(col))
              for i in 1:NU for q in 1:NU for col in 1:NP]
    @add_con!(c, slack_rows, t.id => t.w *
        (t.c1 * gains[t.i, 1, 1] + t.c2 * gains[t.i, 2, 1] + t.c3 * gains[t.i, 3, 1] +
         t.c4 * gains[t.i, 4, 1] + t.c5 * gains[t.i, 5, 1] + t.c6 * gains[t.i, 6, 1]) *
        (t.c1 * gains[t.j, 1, 1] + t.c2 * gains[t.j, 2, 1] + t.c3 * gains[t.j, 3, 1] +
         t.c4 * gains[t.j, 4, 1] + t.c5 * gains[t.j, 5, 1] + t.c6 * gains[t.j, 6, 1]) for t in pair0)
    @add_con!(c, slack_rows, t.id => t.w *
        (t.c1 * gains[t.q, 1, 1] + t.c2 * gains[t.q, 2, 1] + t.c3 * gains[t.q, 3, 1] +
         t.c4 * gains[t.q, 4, 1] + t.c5 * gains[t.q, 5, 1] + t.c6 * gains[t.q, 6, 1])^2 for t in trace0)

    # navigation and jitter part: -alpha K diag(m) K' and, on the diagonal, -beta tr(K diag(m) K')
    nav_pair = [(id = row_of[(a, i, j)], a = a, i = i, j = j, b = b, w = -kappa[a] * alpha * gs2(a) * m[b])
                for a in 1:n for i in 1:NU for j in 1:i for b in 1:NP]
    nav_trace = [(id = row_of[(a, i, i)], a = a, q = q, b = b, w = -kappa[a] * beta * gs2(a) * m[b])
                 for a in 1:n for i in 1:NU for q in 1:NU for b in 1:NP]
    @add_con!(c, slack_rows, t.id => t.w * gains[t.i, t.b, t.a] * gains[t.j, t.b, t.a] for t in nav_pair)
    @add_con!(c, slack_rows, t.id => t.w * gains[t.q, t.b, t.a]^2 for t in nav_trace)

    # Gates execution error -Q_G(f) = -(s4 ||f||^2 I + (s2 - s4) f f')
    gates = [(id = row_of[(r.k, r.i, r.j)], k = r.k, i = r.i, j = r.j, w1 = -kappa[r.k] * p.var_point * (r.i == r.j),
              w2 = -kappa[r.k] * (p.var_mag - p.var_point)) for r in rows]
    @add_con!(c, slack_rows, t.id => t.w1 * (ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2) + t.w2 * ff[t.i, t.k] * ff[t.j, t.k]
        for t in gates)
    return c
end
