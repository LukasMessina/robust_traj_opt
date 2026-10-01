# Stochastic multiple-shooting NLP (the seeded TrajectoryNLP of the JAX script).
#
# Variables (56 N + 28, the reference count): node means 7(N+1), normalized
# feedforward 3N, scaled gains K_hat = K_tilde / gain_scale 18N, seed-scaled node
# Cholesky factors of nodes 1..N 28N, terminal margin factor 21.
# Constraints (36 N + 34): initial state 7, terminal mean 6, mean defects 7N,
# covariance defects 28N, terminal residual 21, thrust chance N.
# Bounds: nonnegative factor diagonals, margin diagonal >= terminal_margin_floor.
#
# Objective per arc k (unweighted, equal arcs):
#   ||S_k||_eps + tr(P_k[1:6,1:6]) / c^2 + tr(T_k),
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
    @add_var(c, gains, 1:NU, 1:NP, 1:n; start = seed.gains ./ reshape(seed.gain_scale, 1, 1, :))
    @add_var(c, chol, 1:NL, 1:n; start = scaling.factors, lvar = lvar_chol)
    @add_var(c, margin, 1:NM; start = seed.terminal_margin, lvar = lvar_margin)

    # ------------------------------------------------------------ objective
    epsilon2 = o.control_norm_eps^2
    @add_obj(c, sqrt(ff[1, k]^2 + ff[2, k]^2 + ff[3, k]^2 + epsilon2) for k in 1:n)
    trace_data = [(k = a - 1, r = r, w = scaling.std[LTRI_ROWS[r], a]^2 * p.inv_sigma_sq) for a in 2:n for r in 1:NM]
    @add_obj(c, t.w * chol[t.r, t.k]^2 for t in trace_data)
    cA = 1 + p.var_mag + 2p.var_point
    cf = p.var_mag + 2p.var_point
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
    c = ExaModels.constraint(c, chance_oracle(n, offsets, 56n + 28, scaling, seed, p, backend))
    model = ExaModel(c)
    model.meta.nvar == 56n + 28 || error("variable count $(model.meta.nvar) differs from the reference 56N + 28")
    model.meta.ncon == 36n + 34 || error("constraint count $(model.meta.ncon) differs from the reference 36N + 34")
    P0 = node_covariance(scaling.initial_factor, scaling.std[:, 1])
    constant = tr(P0[1:NP, 1:NP]) * p.inv_sigma_sq + n * 3p.jitter
    return (; model, means, ff, gains, chol, margin, objective_constant = constant, ut)
end
