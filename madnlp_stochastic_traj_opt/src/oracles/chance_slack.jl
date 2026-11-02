# Cholesky-slack (epigraph) form of the thrust chance constraint for the polar and throttle controls and
# the multiphase layout (chance_formulation = :slack): per controlled arc k the 6 packed lower rows
#
#   tau_hat_k^2 I - G_hat_k G_hat_k' - T_k / s_k^2 = 0,
#
# with T_k the closed form `control_covariance` of the chance rows (the inputs of `chance_layout`: f, K_hat,
# L_tilde rows 1..6 and the throttle) and s_k = sqrt(lambda_max(T_k)) at the start, a fixed scale. G_hat_k is
# lower triangular with diagonal >= chance_slack_floor / s_k, so tau_k = s_k tau_hat_k >= rho(T_k) without an
# eigenvalue: the chance row is Gamma_k + psi s_k tau_hat_k <= 1 and the chance_sum cost
# sum_k w_k (Gamma_k + psi s_k tau_hat_k). Inputs nchance(p) + 1 (tau_hat) and + 2..7 (G_hat, packed row-major
# lower: (1,1), (2,1), (2,2), (3,1), (3,2), (3,3)); exact derivatives from single and pair-seeded duals.

const NSLACK = NU * (NU + 1) ÷ 2                 # 6 rows and G_hat entries per arc
@inline nslack_inputs(p::UTParams) = nchance(p) + 1 + NSLACK
@inline nslack_pairs(p::UTParams) = nslack_inputs(p) * (nslack_inputs(p) + 1) ÷ 2
const SLACK_ROWS = (1, 2, 2, 3, 3, 3)
const SLACK_COLS = (1, 1, 2, 1, 2, 3)

"Lower 3 x 3 matrix of the packed row-major entries `g`."
@inline slack_matrix(g) = SMatrix{3,3}(g[1], g[2], g[4], zero(g[1]), g[3], g[5], zero(g[1]), zero(g[1]), g[6])

"The 6 rows of arc `a` from its inputs (any number type)."
@inline function slack_values(p::UTParams, std, gs, theta, k, L, n, gamma, tau, g, kappa)
    f = feedforward_command(p, theta, n, gamma)
    T = control_covariance(p, f, gs * SMatrix{3,6}(k), position_velocity_covariance(std, L))
    G = slack_matrix(g)
    GG = G * G'
    return SVector{NSLACK}(ntuple(Val(NSLACK)) do r
        i, j = SLACK_ROWS[r], SLACK_COLS[r]
        (i == j ? tau^2 : zero(tau^2)) - GG[i, j] - kappa * T[i, j]
    end)
end

@inline slack_inputs(x, index, fixed, a, nq) = (chance_input(x, index, fixed, nq + 1, a),
    SVector{NSLACK}(ntuple(i -> chance_input(x, index, fixed, nq + 1 + i, a), Val(NSLACK))))

@kernel function slack_value_kernel!(out, @Const(x), @Const(index), @Const(fixed), @Const(arc_std), @Const(gain_scale),
        @Const(kappa), p::UTParams, full)
    a = @index(Global, Linear)
    f, k, L = chance_inputs(x, index, fixed, a)
    tau, g = slack_inputs(x, index, fixed, a, nchance(p))
    v = slack_values(p, start_scale(full, arc_std, a), @inbounds(gain_scale[a]), f, k, L, chance_direction(fixed, a),
        chance_throttle(p, x, index, fixed, a), tau, g, @inbounds(kappa[a]))
    for r in 1:NSLACK
        @inbounds out[NSLACK * (a - 1) + r] = v[r]
    end
end

@kernel function slack_jacobian_kernel!(jac, @Const(x), @Const(index), @Const(fixed), @Const(arc_std), @Const(gain_scale),
        @Const(kappa), p::UTParams, full)
    i = @index(Global, Linear)
    nq, ns = nchance(p), nslack_inputs(p)
    q, a = (i - 1) % ns + 1, (i - 1) ÷ ns + 1
    f, k, L = chance_inputs(x, index, fixed, a)
    tau, g = slack_inputs(x, index, fixed, a, nq)
    fd = SVector{3}(ntuple(j -> single_seed(f[j], q == j), Val(3)))
    kd = SVector{18}(ntuple(j -> single_seed(k[j], q == 3 + j), Val(18)))
    Ld = SVector{21}(ntuple(j -> single_seed(L[j], q == 21 + j), Val(21)))
    gd = throttle_seed(p, chance_throttle(p, x, index, fixed, a), q)
    td = single_seed(tau, q == nq + 1)
    Gd = SVector{NSLACK}(ntuple(j -> single_seed(g[j], q == nq + 1 + j), Val(NSLACK)))
    v = slack_values(p, start_scale(full, arc_std, a), @inbounds(gain_scale[a]), fd, kd, Ld, chance_direction(fixed, a),
        gd, td, Gd, @inbounds(kappa[a]))
    for r in 1:NSLACK
        @inbounds jac[r, q, a] = ForwardDiff.partials(v[r])[1]
    end
end

@kernel function slack_hessian_kernel!(hess, @Const(x), @Const(y), @Const(index), @Const(fixed), @Const(arc_std),
        @Const(gain_scale), @Const(kappa), @Const(pair_p), @Const(pair_q), p::UTParams, full)
    i = @index(Global, Linear)
    nq = nchance(p)
    pair, a = (i - 1) % nslack_pairs(p) + 1, (i - 1) ÷ nslack_pairs(p) + 1
    pp, qq = @inbounds(pair_p[pair]), @inbounds(pair_q[pair])
    f, k, L = chance_inputs(x, index, fixed, a)
    tau, g = slack_inputs(x, index, fixed, a, nq)
    fd = SVector{3}(ntuple(j -> pair_seed(f[j], pp == j, qq == j), Val(3)))
    kd = SVector{18}(ntuple(j -> pair_seed(k[j], pp == 3 + j, qq == 3 + j), Val(18)))
    Ld = SVector{21}(ntuple(j -> pair_seed(L[j], pp == 21 + j, qq == 21 + j), Val(21)))
    gd = throttle_pair_seed(p, chance_throttle(p, x, index, fixed, a), pp, qq)
    td = pair_seed(tau, pp == nq + 1, qq == nq + 1)
    Gd = SVector{NSLACK}(ntuple(j -> pair_seed(g[j], pp == nq + 1 + j, qq == nq + 1 + j), Val(NSLACK)))
    v = slack_values(p, start_scale(full, arc_std, a), @inbounds(gain_scale[a]), fd, kd, Ld, chance_direction(fixed, a),
        gd, td, Gd, @inbounds(kappa[a]))
    total = 0.0
    for r in 1:NSLACK
        total += @inbounds(y[NSLACK * (a - 1) + r]) * pair_value(v[r])
    end
    @inbounds hess[pair, a] = total
end

"""
The slack rows of the arcs in the chance `layout` (`chance_layout`, with the throttle when the controls have one):
the inputs of arc a are its chance inputs, then tau_hat (variable `tau + a`) and G_hat (variables
`factor + NSLACK (a - 1) + (1:NSLACK)`); `kappa[a]` = 1 / s_a^2.
"""
function chance_slack_oracle(layout, nvar::Int, p::UTParams, backend, tau::Int, factor::Int, kappa::AbstractVector)
    (; index, fixed, arc_std, gain_scale) = layout
    full = get(layout, :full, Val(false))
    nq, n = nchance(p), size(index, 2)
    size(index, 1) == nq || error("the chance layout needs $nq inputs")
    ns = nslack_inputs(p)
    slack_index = vcat(index, reshape(tau .+ (1:n), 1, n), factor .+ reshape(1:NSLACK * n, NSLACK, n))
    slack_fixed = vcat(fixed, zeros(1 + NSLACK, n))
    pair_p = [pp for pp in 1:ns for _ in 1:pp]
    pair_q = [qq for pp in 1:ns for qq in 1:pp]
    jac_rows, jac_cols, jac_gather = Int[], Int[], Int[]
    hess_rows, hess_cols, hess_gather = Int[], Int[], Int[]
    for a in 1:n
        for q in 1:ns, r in 1:NSLACK
            slack_index[q, a] == 0 && continue
            push!(jac_rows, NSLACK * (a - 1) + r); push!(jac_cols, slack_index[q, a])
            push!(jac_gather, r + NSLACK * (q - 1) + NSLACK * ns * (a - 1))
        end
        for pp in 1:ns, qq in 1:pp
            ip, iq = slack_index[pp, a], slack_index[qq, a]
            (ip == 0 || iq == 0) && continue
            push!(hess_rows, max(ip, iq)); push!(hess_cols, min(ip, iq))
            push!(hess_gather, lower_position(pp, qq) + nslack_pairs(p) * (a - 1))
        end
    end
    device(v) = ExaModels.convert_array(v, backend)
    di, df = device(slack_index), device(slack_fixed)
    ds, dg, dk = device(Matrix{Float64}(arc_std)), device(Vector{Float64}(gain_scale)), device(Vector{Float64}(kappa))
    dp, dq, djg, dhg = device(pair_p), device(pair_q), device(jac_gather), device(hess_gather)
    jac_buffer, hess_buffer = device(zeros(NSLACK, ns, n)), device(zeros(nslack_pairs(p), n))
    return ExaModels.VectorNonlinearOracle(; nvar, ncon = NSLACK * n, jac_rows, jac_cols, hess_rows, hess_cols,
        lcon = device(zeros(NSLACK * n)), ucon = device(zeros(NSLACK * n)),
        f! = (out, x) -> launch!(slack_value_kernel!, n, out, x, di, df, ds, dg, dk, p, full),
        jac! = function (vals, x)
            launch!(slack_jacobian_kernel!, n * ns, jac_buffer, x, di, df, ds, dg, dk, p, full)
            gather!(vals, jac_buffer, djg)
        end,
        hess! = function (vals, x, y)
            launch!(slack_hessian_kernel!, n * nslack_pairs(p), hess_buffer, x, y, di, df, ds, dg, dk, dp, dq, p, full)
            gather!(vals, hess_buffer, dhg)
        end)
end

"Start of the slack variables: s = sqrt(lambda_max(T)), tau_hat = tau / s and G_hat = chol(tau^2 I - T) / s packed."
function slack_oracle_start(T::AbstractMatrix, floor)
    s, tau_hat, _ = slack_chance_start(T, floor)
    G = cholesky(Symmetric((tau_hat * s)^2 * I - T)).L / s
    return s, tau_hat, [G[SLACK_ROWS[r], SLACK_COLS[r]] for r in 1:NSLACK]
end
