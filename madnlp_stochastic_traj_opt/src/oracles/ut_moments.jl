# UT moment oracle: the propagated node mean and normalized covariance of every
# arc, added (with a minus sign) to the native rows `means[:, k+1]` and
# `pack(L_{k+1} L_{k+1}')` through ExaModels.add_eval. No decision variable is
# introduced.
#
# Arc inputs theta = (m 7, d 3, K_hat 18, L_tilde 28, Gamma): the direction d and the throttle Gamma give the
# feedforward S = Gamma d (`ntheta(p)` = 57 inputs: the GPU chunks and assembly tiles of 8 then have a partial
# last one). An input that is not a variable (the fixed initial factor, the controls of an arc without thrust)
# is read from `theta_fixed`. For sigma point s, z_s = (x_s, u_s) = Z_s(theta) and
# y_s = Phi(z_s) is one XMDS2 RK9 step. With l = a'ybar + tr(B C) and
# g_s = dl/dy_s = w (a + 2 N (y_s - ybar)):
#
#   Hessian = sum_s W_s' [H_s W_s + 2 w J_s' N (V_s - Vbar)] + Hess(zeta)
#
# W_s = dZ_s/dtheta, J_s = dPhi/dz, H_s = Hess_z(g_s' Phi) (forward-over-reverse),
# V_s = J_s W_s, Vbar = sum_s w V_s, zeta(theta) = sum_s (J_s' g_s)' Z_s(theta).
# Every term is exact. The Gauss-Newton term 2 w sum_s V_s' N (V_s - Vbar) equals
# 2 w sum_s (V_s - Vbar)' N (V_s - Vbar) because sum_s w (V_s - Vbar) = 0; with the
# centered right factor its rounding error is relative to |V| |N| |V_s - Vbar|, not to
# |V|^2 |N| as in the uncentered sum_s 2 w V_s' N V_s - 2 Vbar' N Vbar, whose terms agree
# to the relative spread of the sigma points (1e-5 on a heliocentric arc).
# The Hessian also relies on sum_s w (y_s - ybar) = 0 and sum_s w (V_s - Vbar) = 0: a residual
# of the rounded mean multiplies the full point Hessians (respectively J_s' N), an error of
# relative size eps / spread^2. Its deviations therefore use the corrected two-pass mean:
# ybar and Vbar carry the first-pass mean m and the correction r = sum_s w (y_s - m) in
# rows 8..14, and a deviation is (y_s - m) - r, centered to eps |y_s - m|.
# The state spread is linear in L (`spread_factor`), so zeta carries only the
# curvature of the executed controls (bilinear gain map, Gates factors).
#
# The rows written by an oracle are `rows` (local to its constraint blocks) and the
# mean rows are scaled by `ratio`: an arc whose end node has another origin or scale
# than its phase (a frame switch) maps the UT mean to the end node's units there; the
# constant origin translation is in the row bounds. Several oracles (one per set of UT
# parameters, e.g. per phase) may write disjoint rows of the same blocks.
#
# The per-arc kernels are shared by both backends. The per-sigma-point derivatives
# (W_s, J_s, H_s and the products V_s, B_s) are organized per backend:
#  * CPU: one thread per sigma point, with dual numbers over all 57 arc inputs;
#  * GPU: stages split over threads to keep the dual numbers small: W_s in chunks of
#    8 of the 57 directions, J_s and H_s per point, V_s and B_s one column per thread,
#    and sum_s W_s' B_s in shared-memory tiles.
# Every kernel reads `@index(Global, Linear)` in a plain assignment: KernelAbstractions'
# CPU backend does not rewrite it inside a larger expression.

const NUT_ROWS = NX + NL                       # 35 oracle rows per arc

# GPU: dual-number directions per thread of the 57 arc inputs.
const GPU_ARC_CHUNK = Val(8)

struct UTOracleData{VI,MI,VF,MF}
    theta_index::MI      # 57 x n local-x index of each arc input (0: fixed data)
    theta_fixed::MF      # 57 x n values of the fixed inputs
    rows::MI             # 35 x n output rows, local to the oracle's constraint blocks
    ratio::MF            # 7 x n mean-row factor: end-node units per propagated unit
    arc_std::MF          # start node scale: 7 x n standard deviations, or 28 x n packed lower S_k (full)
    inv_next::MF         # end node inverse scale: 7 x n, or 28 x n packed lower S_{k+1}^-1 (full)
    gain_scale::VF
    steps::VF
    pair_p::VI           # 1596 dense lower-triangle pairs
    pair_q::VI
    jac_gather::VI
    hess_gather::VI
end

struct UTWorkspace{A2,A3,A4}
    Y::A3        # 7 x P x n propagated sigma points
    Dn::A3       # 7 x P x n normalized deviations of the sigma points
    ybar::A2     # 14 x n: mean (rows 1..7) and its two-pass correction (rows 8..14)
    W::A4        # 10 x 56 x P x n  W_s = dz_s/dtheta
    V::A4        # 7 x 56 x P x n  V_s = J_s W_s
    B::A4        # 10 x 56 x P x n  B_s = H_s W_s + 2 w J_s' N (V_s - Vbar)
    mu::A3       # 10 x P x n  J_s' g_s
    Jpt::A4      # 7 x 10 x P x n  J_s
    Vbar::A3     # 14 x 56 x n: sum_s w V_s and its two-pass correction
    aN::A2       # 56 x n: a (7) and N (7x7, column-major)
    jac::A3      # 35 x 56 x n
    gH::A3       # 9 x 2d x n  command gradients and packed Hessians of the outer points
    zeta::A2     # 1596 x n
    hess::A2     # 1596 x n
    # GPU stages only (empty on the CPU)
    Z::A3        # 10 x P x n sigma-point inputs z_s
    Hraw::A4     # 10 x 10 x P x n  forward-over-reverse Hessian columns
    M::A4        # 10 x 10 x P x n  symmetrized H_s
end

struct UTOracle{D<:UTOracleData,W<:UTWorkspace,S,P<:UTParams,F}
    data::D
    workspace::W
    structure::S         # jac_rows, jac_cols, hess_rows, hess_cols (local indices)
    params::P
    n::Int
    gpu::Bool
    full::F              # Val(true): full lower node scales, Val(false): diagonal
end

# ------------------------------------------------------------ sigma points

@inline function arc_input(x, theta_index, theta_fixed, q, a)
    i = @inbounds theta_index[q, a]
    return i == 0 ? @inbounds(theta_fixed[q, a]) : @inbounds(x[i])
end

@inline function load_arc(x, theta_index, theta_fixed, a)
    m = SVector{7}(ntuple(i -> arc_input(x, theta_index, theta_fixed, i, a), Val(7)))
    f = SVector{3}(ntuple(i -> arc_input(x, theta_index, theta_fixed, 7 + i, a), Val(3)))
    k = SVector{18}(ntuple(i -> arc_input(x, theta_index, theta_fixed, 10 + i, a), Val(18)))
    L = SVector{28}(ntuple(i -> arc_input(x, theta_index, theta_fixed, 28 + i, a), Val(28)))
    return m, f, k, L
end

"Throttle Gamma of an arc (input 57)."
@inline arc_throttle(::UTParams, x, theta_index, theta_fixed, a) = arc_input(x, theta_index, theta_fixed, 57, a)

"The arc's feedforward S = Gamma d from its loaded direction inputs `theta`."
@inline arc_feedforward(p::UTParams, x, theta_index, theta_fixed, theta, a) = arc_throttle(p, x, theta_index, theta_fixed, a) * theta

@inline arc_column(A, a) = SVector{7}(ntuple(i -> @inbounds(A[i, a]), Val(7)))
@inline packed_scale(A, a) = lower_matrix(SVector{28}(ntuple(i -> @inbounds(A[i, a]), Val(28))))

# Node scales of an arc: diagonal (SVector of standard deviations or their inverses) or full
# (lower SMatrix S_k or S_{k+1}^-1). Diagonal end scales enter the covariance rows after the
# moment sums (the original arithmetic); full ones map each deviation before them.
@inline start_scale(::Val{false}, A, a) = arc_column(A, a)
@inline start_scale(::Val{true}, A, a) = packed_scale(A, a)
@inline end_scale(::Val{false}, A, a) = arc_column(A, a)
@inline end_scale(::Val{true}, A, a) = packed_scale(A, a)
@inline end_deviation(e, ::SVector{7}) = e
@inline end_deviation(e, Sinv::SMatrix{7,7}) = Sinv * e
@inline scaled_row(total, inv::SVector{7}, i, j) = -(total * inv[i] * inv[j])
@inline scaled_row(total, ::SMatrix{7,7}, i, j) = -total
@inline row_weight(lambda, inv::SVector{7}, i, j) = -lambda * inv[i] * inv[j]
@inline row_weight(lambda, ::SMatrix{7,7}, i, j) = -lambda
@inline end_weight(B, ::SVector{7}) = B
@inline end_weight(B, Sinv::SMatrix{7,7}) = Sinv' * B * Sinv

"Outer sigma point `o` (1..2d): state, commanded control c_o = S + sigma K m_o and Wiener increment."
@inline function outer_point(p::UTParams, m::SVector{7}, f::SVector{3}, K::SMatrix{3,6}, S::SVector{28}, o::Int, h)
    TS = eltype(S)
    j = (o - 1) % p.d + 1
    sigma = o <= p.d ? 1.0 : -1.0
    column = j <= NX ? packed_column(S, j) : zero(SVector{7,TS})
    nav = j - NX
    measurement = j <= NX ? SVector{6,TS}(ntuple(i -> column[i], Val(6))) :
        j <= NX + NP ? SVector{6,TS}(ntuple(i -> TS(p.nav_factor[i, nav]), Val(6))) : zero(SVector{6,TS})
    x = m + sigma * to_state(p.scale, column)
    c = f + sigma * (K * measurement)
    return x, c, wiener_increment(p, o, h)
end

"Wiener increment of outer sigma point `o`."
@inline function wiener_increment(p::UTParams, o::Int, h)
    j = (o - 1) % p.d + 1
    sigma = o <= p.d ? 1.0 : -1.0
    wiener = j - NX - NP
    return SVector{3,Float64}(ntuple(i -> i == wiener ? sqrt(h) * (sigma * p.sqrt_d) : 0.0, Val(3)))
end

"Executed control of inner Gates point `l` (1..6) given the outer command."
@inline function inner_control(p::UTParams, c::SVector{3}, G::SMatrix{3,3}, l::Int)
    offset = p.sqrt3 * G[:, (l - 1) % 3 + 1]
    e = l <= 3 ? c + offset : c + (-offset)
    return p.tmax * e
end

@inline outer_index(p::UTParams, s::Int) = (s - 1) ÷ p.inner + 1

"Executed control of sigma point `s` given its outer command (the command itself without Gates points)."
@inline executed_control(p::UTParams, c::SVector{3}, s::Int) =
    p.inner == 1 ? p.tmax * c : inner_control(p, c, gates_factor(c, p), (s - 1) % p.inner + 1)

"Sigma point `s` = (outer point, executed point): state, executed control, Wiener increment."
@inline function point_input(p::UTParams, m, f, K, S, s::Int, h)
    x, c, dw = outer_point(p, m, f, K, S, outer_index(p, s), h)
    return x, executed_control(p, c, s), dw
end

@inline sigma_step(p::UTParams, x, u, dw, h, s) = rk_step(XMDS2RK9, x, u,
    sigma_mass_flow(p, u, outer_flow_sq(p, outer_index(p, s))), diffusion_increment(p.sigma_nd, dw), h, p.gravity, p.ve)

@inline point_state(z) = SVector{7}(ntuple(i -> z[i], Val(7)))
@inline point_control(z) = SVector{3}(ntuple(i -> z[7+i], Val(3)))
@inline point_jacobian(y) = SMatrix{7,10,Float64}(ntuple(l -> partial(y[(l-1)%7+1], (l-1)÷7+1), Val(70)))
@inline load_N(aN, a) = SMatrix{7,7,Float64}(ntuple(i -> @inbounds(aN[7+i, a]), Val(49)))
@inline load_point_jacobian(Jpt, s, a) =
    SMatrix{7,10,Float64}(ntuple(l -> @inbounds(Jpt[(l-1)%7+1, (l-1)÷7+1, s, a]), Val(70)))

"Weighted mean of component `i` over the sigma points."
@inline function sigma_mean(Y, i, a, p::UTParams)
    total = 0.0
    for s in 1:p.n_points
        total += p.w_point * @inbounds(Y[i, s, a])
    end
    return total
end

"Correction sum_s w (Y_s - m) of the first-pass mean m of component `i` (corrected two-pass mean)."
@inline function sigma_mean_correction(Y, i, a, m, p::UTParams)
    total = 0.0
    for s in 1:p.n_points
        total += p.w_point * (@inbounds(Y[i, s, a]) - m)
    end
    return total
end

"Centered deviation of sigma point `s`: (y_s - m) - r with the two-pass mean of `ybar`."
@inline point_deviation(Y, ybar, s, a) =
    SVector{7}(ntuple(i -> (@inbounds(Y[i, s, a]) - @inbounds(ybar[i, a])) - @inbounds(ybar[7+i, a]), Val(7)))

"""
Forward-over-reverse RK9 step of sigma point `s` from its input `z`: the propagated
point `y` and `zb = J_s' g_s`, both as duals over the 10 point inputs.
"""
@inline function point_adjoint(p::UTParams, z, dw, Y, ybar, aN, s, a, h)
    N = load_N(aN, a)
    av = SVector{7}(ntuple(i -> @inbounds(aN[i, a]), Val(7)))
    g = p.w_point * (av + 2 * (N * point_deviation(Y, ybar, s, a)))
    xd, ud = seed(TagPoint, point_state(z), 0, Val(10)), seed(TagPoint, point_control(z), 7, Val(10))
    magnitude = sigma_mass_flow(p, ud, outer_flow_sq(p, outer_index(p, s)))
    gdw = diffusion_increment(p.sigma_nd, dw)
    y, stages = rk_step_stages(XMDS2RK9, xd, ud, magnitude, gdw, h, p.gravity, p.ve)
    xb, ub, mb = rk_adjoint(XMDS2RK9, stages, ud, magnitude, g, h, p.gravity, p.ve)
    # Without thrust the mass flow is identically zero (no d|u|/du term).
    return y, vcat(xb, p.tmax == 0 ? ub : ub + mb * (ud / magnitude))
end

# ------------------------------------------------------------------- values

@kernel function ut_forward_kernel!(Y, @Const(x), @Const(theta_index), @Const(theta_fixed), @Const(arc_std),
        @Const(gain_scale), @Const(steps), p::UTParams, full)
    index = @index(Global, Linear)
    a, s = (index - 1) ÷ p.n_points + 1, (index - 1) % p.n_points + 1
    m, theta, k, L = load_arc(x, theta_index, theta_fixed, a)
    f = arc_feedforward(p, x, theta_index, theta_fixed, theta, a)
    h = @inbounds steps[a]
    S = spread_factor(p, start_scale(full, arc_std, a), L)
    K = @inbounds(gain_scale[a]) * SMatrix{3,6}(k)
    xs, us, dw = point_input(p, m, f, K, S, s, h)
    y = sigma_step(p, xs, us, dw, h, s)
    for i in 1:7
        @inbounds Y[i, s, a] = y[i]
    end
end

"Mean rows (also stored in ybar)."
@kernel function ut_mean_kernel!(out, ybar, @Const(Y), @Const(rows), @Const(ratio), p::UTParams)
    index = @index(Global, Linear)
    a, row = (index - 1) ÷ NX + 1, (index - 1) % NX + 1
    m = sigma_mean(Y, row, a, p)
    @inbounds ybar[row, a] = m
    @inbounds out[rows[row, a]] = -(ratio[row, a] * m)
end

"Normalized deviations of the sigma points from ybar."
@kernel function ut_deviation_kernel!(Dn, @Const(Y), @Const(ybar), @Const(inv_next), p::UTParams, full)
    index = @index(Global, Linear)
    s, a = (index - 1) % p.n_points + 1, (index - 1) ÷ p.n_points + 1
    e = to_normalized(p.scale, SVector{7}(ntuple(i -> @inbounds(Y[i, s, a]) - @inbounds(ybar[i, a]), Val(7))))
    e = end_deviation(e, end_scale(full, inv_next, a))
    for i in 1:7
        @inbounds Dn[i, s, a] = e[i]
    end
end

"Scaled covariance rows from the normalized deviations."
@kernel function ut_covariance_kernel!(out, @Const(Dn), @Const(rows), @Const(inv_next), p::UTParams, full)
    index = @index(Global, Linear)
    a, r = (index - 1) ÷ NL + 1, (index - 1) % NL + 1
    i, j = LTRI_ROWS[r], LTRI_COLS[r]
    total = 0.0
    for s in 1:p.n_points
        total += (p.w_point * @inbounds(Dn[i, s, a])) * @inbounds(Dn[j, s, a])
    end
    @inbounds out[rows[NX + r, a]] = scaled_row(total, end_scale(full, inv_next, a), i, j)
end

# ------------------------------------------------- per-arc derivative kernels

"Sigma-point mean (q = 0) and Vbar = sum_s w V_s (q = 1..56), each with its two-pass correction."
@kernel function ut_reduce_kernel!(ybar, Vbar, @Const(Y), @Const(V), p::UTParams)
    index = @index(Global, Linear)
    i = (index - 1) % 7 + 1
    q = ((index - 1) ÷ 7) % (ntheta(p) + 1)
    a = (index - 1) ÷ (7 * (ntheta(p) + 1)) + 1
    if q == 0
        m = sigma_mean(Y, i, a, p)
        @inbounds ybar[i, a] = m
        @inbounds ybar[7+i, a] = sigma_mean_correction(Y, i, a, m, p)
    else
        total = 0.0
        for s in 1:p.n_points
            total += p.w_point * @inbounds(V[i, q, s, a])
        end
        correction = 0.0
        for s in 1:p.n_points
            correction += p.w_point * (@inbounds(V[i, q, s, a]) - total)
        end
        @inbounds Vbar[i, q, a] = total
        @inbounds Vbar[7+i, q, a] = correction
    end
end

"Normalized deviations: Dn = (y_s - ybar) (q = 0) and, in place, V_s - Vbar (q = 1..56)."
@kernel function ut_normalize_kernel!(V, Dn, @Const(Y), @Const(ybar), @Const(Vbar), @Const(inv_next), p::UTParams, full)
    index = @index(Global, Linear)
    q = (index - 1) % (ntheta(p) + 1)
    point = (index - 1) ÷ (ntheta(p) + 1)
    s, a = point % p.n_points + 1, point ÷ p.n_points + 1
    finish = end_scale(full, inv_next, a)
    if q == 0
        e = end_deviation(to_normalized(p.scale, SVector{7}(ntuple(i -> @inbounds(Y[i, s, a]) - @inbounds(ybar[i, a]), Val(7)))), finish)
        for i in 1:7
            @inbounds Dn[i, s, a] = e[i]
        end
    else
        e = end_deviation(to_normalized(p.scale, SVector{7}(ntuple(i -> @inbounds(V[i, q, s, a]) - @inbounds(Vbar[i, q, a]), Val(7)))), finish)
        for i in 1:7
            @inbounds V[i, q, s, a] = e[i]
        end
    end
end

"Jacobian rows from Vbar (means) and the normalized deviations (covariances; V holds V_s - Vbar normalized)."
@kernel function ut_jacobian_kernel!(jac, @Const(Dn), @Const(V), @Const(Vbar), @Const(ratio), @Const(inv_next), p::UTParams, full)
    index = @index(Global, Linear)
    row = (index - 1) % NUT_ROWS + 1
    q = ((index - 1) ÷ NUT_ROWS) % ntheta(p) + 1
    a = (index - 1) ÷ (NUT_ROWS * ntheta(p)) + 1
    if row <= NX
        @inbounds jac[row, q, a] = -(ratio[row, a] * Vbar[row, q, a])
    else
        r = row - NX
        i, j = LTRI_ROWS[r], LTRI_COLS[r]
        total = 0.0
        for s in 1:p.n_points
            di, dj = @inbounds(Dn[i, s, a]), @inbounds(Dn[j, s, a])
            ddi, ddj = @inbounds(V[i, q, s, a]), @inbounds(V[j, q, s, a])
            total += p.w_point * (ddi * dj + di * ddj)
        end
        @inbounds jac[row, q, a] = scaled_row(total, end_scale(full, inv_next, a), i, j)
    end
end

"""
Sigma-point mean and the multiplier blocks: a = -ratio lambda_mean and the weight B of
state deviations with tr(B C) = -sum lambda_r inv_i inv_j C_ij (normalized by `ut_weight_kernel!`).
"""
@kernel function ut_hessian_prep_kernel!(ybar, aN, @Const(Y), @Const(y), @Const(rows), @Const(ratio),
        @Const(inv_next), p::UTParams, full)
    index = @index(Global, Linear)
    k, a = (index - 1) % NUT_ROWS + 1, (index - 1) ÷ NUT_ROWS + 1
    if k <= NX
        m = sigma_mean(Y, k, a, p)
        @inbounds ybar[k, a] = m
        @inbounds ybar[7+k, a] = sigma_mean_correction(Y, k, a, m, p)
        @inbounds aN[k, a] = -(y[rows[k, a]] * ratio[k, a])
    else
        r = k - NX
        i, j = LTRI_ROWS[r], LTRI_COLS[r]
        lambda = @inbounds y[rows[k, a]]
        value = row_weight(lambda, end_scale(full, inv_next, a), i, j)
        value = i == j ? value : value / 2
        @inbounds aN[7 + i + 7(j-1), a] = value
        @inbounds aN[7 + j + 7(i-1), a] = value
    end
end

"N = T^-T B T^-1: the multiplier weight of state deviations from that of normalized ones."
@kernel function ut_weight_kernel!(aN, @Const(inv_next), p::UTParams, full)
    a = @index(Global, Linear)
    N = normalized_weight(p.scale, end_weight(load_N(aN, a), end_scale(full, inv_next, a)))
    for i in 1:49
        @inbounds aN[7 + i, a] = N[i]
    end
end

"Centered Gauss-Newton term of column q: B_s[:, q] += 2 w J_s' N (V_s - Vbar)[:, q]."
@kernel function ut_gauss_newton_kernel!(B, @Const(V), @Const(Vbar), @Const(Jpt), @Const(aN), p::UTParams)
    index = @index(Global, Linear)
    q, point = (index - 1) % ntheta(p) + 1, (index - 1) ÷ ntheta(p)
    s, a = point % p.n_points + 1, point ÷ p.n_points + 1
    deviation = SVector{7}(ntuple(i -> (@inbounds(V[i, q, s, a]) - @inbounds(Vbar[i, q, a])) - @inbounds(Vbar[7+i, q, a]), Val(7)))
    g = load_point_jacobian(Jpt, s, a)' * ((2 * p.w_point) * (load_N(aN, a) * deviation))
    for r in 1:10
        @inbounds B[r, q, s, a] += g[r]
    end
end

# zeta(theta) = sum_s mu_s' Z_s(theta). The sigma states are linear in theta, and the
# executed controls depend on theta only through the outer commands c_o(theta): with
# phi_o(c) = sum_{s in o} mu_s[8:10]' u_s(c), Hess(zeta) = sum_o Jc_o' H_o Jc_o + g_o . Hess(c_o),
# g_o and H_o the gradient and Hessian of phi_o at c_o. Jc_o = dc_o/dtheta is sparse and closed
# form (identity in f, the measurement in K_hat, the gains in L), and Hess(c_o) has only the
# K_hat-L cross terms of the bilinear map c_o = f + sigma gs K_hat m_o(L). Without thrust zeta = 0.

"Doubly seeded command for the gradient and Hessian of a function of c (3 directions)."
@inline function command_seed(c::SVector{3,Float64})
    return SVector{3}(ntuple(Val(3)) do i
        inner = ForwardDiff.Dual{TagInner}(c[i], unit_partials(Val(3), i))
        shift = ntuple(k -> ForwardDiff.Dual{TagInner}(Float64(k == i), ForwardDiff.Partials{3,Float64}((0.0, 0.0, 0.0))), Val(3))
        ForwardDiff.Dual{TagOuter}(inner, ForwardDiff.Partials{3,typeof(inner)}(shift))
    end)
end

"Gradient g_o (rows 1..3) and packed lower Hessian H_o (rows 4..9) of phi_o at the command of outer point o."
@kernel function ut_command_curvature_kernel!(gH, @Const(mu), @Const(x), @Const(theta_index), @Const(theta_fixed),
        @Const(arc_std), @Const(gain_scale), @Const(steps), p::UTParams, full)
    index = @index(Global, Linear)
    o, a = (index - 1) % (2p.d) + 1, (index - 1) ÷ (2p.d) + 1
    m, theta, k, L = load_arc(x, theta_index, theta_fixed, a)
    f = arc_feedforward(p, x, theta_index, theta_fixed, theta, a)
    S = spread_factor(p, start_scale(full, arc_std, a), L)
    _, c, _ = outer_point(p, m, f, @inbounds(gain_scale[a]) * SMatrix{3,6}(k), S, o, @inbounds(steps[a]))
    if p.inner == 1
        # phi_o = tmax mu_o[8:10]' c: linear
        for i in 1:3
            @inbounds gH[i, o, a] = p.tmax * mu[7+i, o, a]
        end
        for r in 4:9
            @inbounds gH[r, o, a] = 0.0
        end
    else
        cd = command_seed(c)
        G = gates_factor(cd, p)
        total = zero(eltype(cd))
        for l in 1:p.inner
            us = inner_control(p, cd, G, l)
            for i in 1:3
                total += @inbounds(mu[7+i, p.inner * (o - 1) + l, a]) * us[i]
            end
        end
        outer = ForwardDiff.partials(total)
        for i in 1:3
            @inbounds gH[i, o, a] = ForwardDiff.value(outer[i])
        end
        @inbounds gH[4, o, a] = ForwardDiff.partials(outer[1])[1]
        @inbounds gH[5, o, a] = (ForwardDiff.partials(outer[2])[1] + ForwardDiff.partials(outer[1])[2]) / 2
        @inbounds gH[6, o, a] = ForwardDiff.partials(outer[2])[2]
        @inbounds gH[7, o, a] = (ForwardDiff.partials(outer[3])[1] + ForwardDiff.partials(outer[1])[3]) / 2
        @inbounds gH[8, o, a] = (ForwardDiff.partials(outer[3])[2] + ForwardDiff.partials(outer[2])[3]) / 2
        @inbounds gH[9, o, a] = ForwardDiff.partials(outer[3])[3]
    end
end

"Entry `col` of the feedback measurement of outer column j: the spread (state), navigation root or zero."
@inline function command_measurement(p::UTParams, j, col, std, L)
    if j <= NX
        return col >= j ? p.sqrt_d * std[col] * L[packed_index(col, j)] : 0.0
    elseif j <= NX + NP
        return p.nav_factor[col, j - NX]
    end
    return 0.0
end

"""
Column `q` (an f, K_hat, L or throttle input) of dc_o/dtheta for outer column j and sign sigma;
Jf = dS/dtheta_f and Jg = dS/dGamma (throttle input 57).
"""
@inline function command_column(p::UTParams, q, j, sigma, gs, std, k, L, Jf, Jg)
    q > 56 && return q == 57 ? Jg : zero(SVector{3,Float64})
    if q <= 10
        return Jf[:, q - 7]
    elseif q <= 28
        i, col = (q - 11) % 3 + 1, (q - 11) ÷ 3 + 1
        value = sigma * gs * command_measurement(p, j, col, std, L)
        return SVector{3}(ntuple(t -> t == i ? value : 0.0, Val(3)))
    end
    r = q - 28
    row = LTRI_ROWS[r]
    (LTRI_COLS[r] == j && row <= NP) || return zero(SVector{3,Float64})
    scale = sigma * gs * p.sqrt_d * std[row]
    return SVector{3}(ntuple(t -> scale * k[t + 3(row - 1)], Val(3)))
end

"g_o . d^2 c_o / d theta_pp d theta_qq (pp >= qq): the K_hat[i, row] - L[row, j] cross terms."
@inline function command_cross(p::UTParams, pp, qq, j, sigma, gs, std, g)
    (28 < pp <= 56 && 10 < qq <= 28) || return 0.0
    r = pp - 28
    row = LTRI_ROWS[r]
    (LTRI_COLS[r] == j && row <= NP && (qq - 11) ÷ 3 + 1 == row) || return 0.0
    return g[(qq - 11) % 3 + 1] * sigma * gs * p.sqrt_d * std[row]
end

"Measurement entry `col` of outer column j with a full lower start scale S: the spread (S L) column."
@inline function command_measurement(p::UTParams, j, col, S::SMatrix{7,7}, L)
    if j <= NX
        col >= j || return 0.0
        total = 0.0
        for m in j:col
            total += S[col, m] * L[packed_index(m, j)]
        end
        return p.sqrt_d * total
    elseif j <= NX + NP
        return p.nav_factor[col, j - NX]
    end
    return 0.0
end

"Column `q` of dc_o/dtheta with a full lower start scale: L[m, j] reaches the command through rows m..6 of S."
@inline function command_column(p::UTParams, q, j, sigma, gs, S::SMatrix{7,7}, k, L, Jf, Jg)
    q > 56 && return q == 57 ? Jg : zero(SVector{3,Float64})
    if q <= 28
        q <= 10 && return Jf[:, q - 7]
        i, col = (q - 11) % 3 + 1, (q - 11) ÷ 3 + 1
        value = sigma * gs * command_measurement(p, j, col, S, L)
        return SVector{3}(ntuple(t -> t == i ? value : 0.0, Val(3)))
    end
    r = q - 28
    m = LTRI_ROWS[r]
    (LTRI_COLS[r] == j && m <= NP) || return zero(SVector{3,Float64})
    total = zero(SVector{3,Float64})
    for row in m:NP
        total += S[row, m] * SVector{3}(k[1 + 3(row - 1)], k[2 + 3(row - 1)], k[3 + 3(row - 1)])
    end
    return (sigma * gs * p.sqrt_d) * total
end

"g_o . d^2 c_o / d K_hat[i, row] d L[m, j] = g_i sigma gs sqrt(d) S[row, m] (row >= m) with a full start scale."
@inline function command_cross(p::UTParams, pp, qq, j, sigma, gs, S::SMatrix{7,7}, g)
    (28 < pp <= 56 && 10 < qq <= 28) || return 0.0
    r = pp - 28
    m = LTRI_ROWS[r]
    row = (qq - 11) ÷ 3 + 1
    (LTRI_COLS[r] == j && m <= NP && row >= m) || return 0.0
    return g[(qq - 11) % 3 + 1] * sigma * gs * p.sqrt_d * S[row, m]
end

# Feedforward map S = Gamma d in c_o = S + sigma gs K_hat m_o: dS/dd and dS/dGamma, and the feedforward pairs'
# g . d^2 S (summed over the outer points, the same map for all of them): d^2 (Gamma d) / dGamma dd_i = e_i.
@inline feedforward_jacobian(theta, gamma) = gamma * SMatrix{3,3}(1.0I), theta
@inline add_feedforward_curvature(total, pp, qq, g) = pp == 57 && 8 <= qq <= 10 ? total + g[qq - 7] : total

"Hess(zeta) along one input pair from the command curvatures of the outer points."
@kernel function ut_zeta_kernel!(zeta, @Const(gH), @Const(x), @Const(theta_index), @Const(theta_fixed),
        @Const(arc_std), @Const(gain_scale), @Const(pair_p), @Const(pair_q), p::UTParams, full)
    index = @index(Global, Linear)
    pair = (index - 1) % nhess(p) + 1
    a = (index - 1) ÷ nhess(p) + 1
    pp, qq = @inbounds(pair_p[pair]), @inbounds(pair_q[pair])
    if p.tmax == 0 || pp <= 7 || qq <= 7 || @inbounds(theta_index[pp, a]) == 0 || @inbounds(theta_index[qq, a]) == 0
        @inbounds zeta[pair, a] = 0.0
    else
        _, theta, k, L = load_arc(x, theta_index, theta_fixed, a)
        std = start_scale(full, arc_std, a)
        gs = @inbounds gain_scale[a]
        # c_o = Gamma d + sigma gs K_hat m_o, so dc_o/dd = Jf, dc_o/dGamma = Jg, and every outer point adds
        # g_o . d^2 S to the feedforward pairs.
        Jf, Jg = feedforward_jacobian(theta, arc_throttle(p, x, theta_index, theta_fixed, a))
        total = 0.0
        gsum = zero(SVector{3,Float64})
        for o in 1:2p.d
            j = (o - 1) % p.d + 1
            sigma = o <= p.d ? 1.0 : -1.0
            u = command_column(p, pp, j, sigma, gs, std, k, L, Jf, Jg)
            v = command_column(p, qq, j, sigma, gs, std, k, L, Jf, Jg)
            g = SVector{3}(ntuple(i -> @inbounds(gH[i, o, a]), Val(3)))
            h11, h21, h22 = @inbounds(gH[4, o, a]), @inbounds(gH[5, o, a]), @inbounds(gH[6, o, a])
            h31, h32, h33 = @inbounds(gH[7, o, a]), @inbounds(gH[8, o, a]), @inbounds(gH[9, o, a])
            H = SMatrix{3,3}(h11, h21, h31, h21, h22, h32, h31, h32, h33)
            total += dot(u, H * v) + command_cross(p, pp, qq, j, sigma, gs, std, g)
            gsum += g
        end
        @inbounds zeta[pair, a] = add_feedforward_curvature(total, pp, qq, gsum)
    end
end

# ------------------------------------------------ CPU: one thread per point

"""
The spread S = sqrt(d) diag(std) L as duals seeded on the factor directions 28 + r within
c0 + 1 : c0 + C: the values of `spread_factor` and its exact derivative sqrt(d) std_row,
without multiplying seeded duals.
"""
@inline function spread_duals(::Type{Tag}, p::UTParams, std, L::SVector{28,Float64}, c0, ::Val{C}) where {Tag,C}
    return SVector{28}(ntuple(Val(28)) do r
        scale = p.sqrt_d * std[LTRI_ROWS[r]]
        ForwardDiff.Dual{Tag}(scale * L[r], ForwardDiff.Partials{C,Float64}(ntuple(j -> c0 + j == 28 + r ? scale : 0.0, Val(C))))
    end)
end

"""
The spread sqrt(d) S L of a full lower start scale S as duals on the factor directions
within c0 + 1 : c0 + C, with its exact constant partials sqrt(d) S[r, m].
"""
@inline function spread_duals(::Type{Tag}, p::UTParams, S::SMatrix{7,7,Float64}, L::SVector{28,Float64}, c0,
        ::Val{C}) where {Tag,C}
    return SVector{28}(ntuple(Val(28)) do index
        r, c = LTRI_ROWS[index], LTRI_COLS[index]
        value = 0.0
        for m in c:r
            value += S[r, m] * L[packed_index(m, c)]
        end
        ForwardDiff.Dual{Tag}(p.sqrt_d * value, ForwardDiff.Partials{C,Float64}(ntuple(Val(C)) do j
            q = c0 + j - 28
            (1 <= q <= NL && LTRI_COLS[q] == c && c <= LTRI_ROWS[q] <= r) ? p.sqrt_d * S[r, LTRI_ROWS[q]] : 0.0
        end))
    end)
end

"The throttle as a dual on direction 57."
@inline throttle_dual(p::UTParams, x, theta_index, theta_fixed, a, nt::Val) =
    seed(TagArc, SVector(arc_throttle(p, x, theta_index, theta_fixed, a)), 56, nt)[1]

"Arc inputs as ntheta(p)-direction duals, the factor as the spread."
@inline function arc_duals(p::UTParams, x, theta_index, theta_fixed, arc_std, a, full)
    m, theta, k, L = load_arc(x, theta_index, theta_fixed, a)
    nt = Val(ntheta(p))
    gamma = throttle_dual(p, x, theta_index, theta_fixed, a, nt)
    return seed(TagArc, m, 0, nt),
        gamma * seed(TagArc, theta, 7, nt),
        seed(TagArc, k, 10, nt), spread_duals(TagArc, p, start_scale(full, arc_std, a), L, 0, nt)
end

"W_s = dz_s/dtheta from the input duals (10 x ntheta)."
@inline function input_jacobian(xs, us, ::Val{NT}) where {NT}
    return SMatrix{10,NT,Float64}(ntuple(Val(10 * NT)) do linear
        r, q = (linear - 1) % 10 + 1, (linear - 1) ÷ 10 + 1
        partial(r <= 7 ? xs[r] : us[r-7], q)
    end)
end

"Sigma-point input z_s, W_s = dz_s/dtheta over all 57 directions, and the Wiener increment."
@inline function point_input_jacobian(p::UTParams, x, theta_index, theta_fixed, arc_std, gain_scale, a, s, h, full)
    md, fd, kd, S = arc_duals(p, x, theta_index, theta_fixed, arc_std, a, full)
    K = @inbounds(gain_scale[a]) * SMatrix{3,6}(kd)
    xs, us, dw = point_input(p, md, fd, K, S, s, h)
    z = SVector{10}(ntuple(i -> ForwardDiff.value(i <= 7 ? xs[i] : us[i-7]), Val(10)))
    return z, input_jacobian(xs, us, Val(ntheta(p))), dw
end

"Propagated sigma points y_s and V_s = J_s W_s."
@kernel function ut_point_jacobian_kernel!(Y, V, @Const(x), @Const(theta_index), @Const(theta_fixed),
        @Const(arc_std), @Const(gain_scale), @Const(steps), p::UTParams, full)
    index = @index(Global, Linear)
    a, s = (index - 1) ÷ p.n_points + 1, (index - 1) % p.n_points + 1
    h = @inbounds steps[a]
    z, W, dw = point_input_jacobian(p, x, theta_index, theta_fixed, arc_std, gain_scale, a, s, h, full)
    y = sigma_step(p, seed(TagPoint, point_state(z), 0, Val(10)), seed(TagPoint, point_control(z), 7, Val(10)), dw, h, s)
    VJ = point_jacobian(y) * W
    for i in 1:7
        @inbounds Y[i, s, a] = ForwardDiff.value(y[i])
        for q in 1:ntheta(p)
            @inbounds V[i, q, s, a] = VJ[i, q]
        end
    end
end

"W_s, B_s = H_s W_s, V_s = J_s W_s, J_s and mu_s = J_s' g_s."
@kernel function ut_point_hessian_kernel!(W_out, B_out, V_out, J_out, mu_out, @Const(Y), @Const(ybar), @Const(aN),
        @Const(x), @Const(theta_index), @Const(theta_fixed), @Const(arc_std), @Const(gain_scale),
        @Const(steps), p::UTParams, full)
    index = @index(Global, Linear)
    a, s = (index - 1) ÷ p.n_points + 1, (index - 1) % p.n_points + 1
    h = @inbounds steps[a]
    z, W, dw = point_input_jacobian(p, x, theta_index, theta_fixed, arc_std, gain_scale, a, s, h, full)
    y, zb = point_adjoint(p, z, dw, Y, ybar, aN, s, a, h)
    J = point_jacobian(y)
    H = SMatrix{10,10,Float64}(ntuple(Val(100)) do linear
        r, q = (linear - 1) % 10 + 1, (linear - 1) ÷ 10 + 1
        (partial(zb[r], q) + partial(zb[q], r)) / 2
    end)
    BW = H * W
    VJ = J * W
    for q in 1:ntheta(p)
        for r in 1:10
            @inbounds W_out[r, q, s, a] = W[r, q]
            @inbounds B_out[r, q, s, a] = BW[r, q]
        end
        for i in 1:7
            @inbounds V_out[i, q, s, a] = VJ[i, q]
        end
    end
    for r in 1:10
        @inbounds mu_out[r, s, a] = ForwardDiff.value(zb[r])
        for i in 1:7
            @inbounds J_out[i, r, s, a] = J[i, r]
        end
    end
end

"Hessian assembly, one lower-triangle entry per thread: sum_s (W_s' B_s)[pp, qq] + zeta."
@kernel function ut_hessian_assemble_kernel!(hess, @Const(W), @Const(B), @Const(zeta), @Const(pair_p),
        @Const(pair_q), p::UTParams)
    index = @index(Global, Linear)
    pair = (index - 1) % nhess(p) + 1
    a = (index - 1) ÷ nhess(p) + 1
    pp, qq = @inbounds(pair_p[pair]), @inbounds(pair_q[pair])
    total = 0.0
    for s in 1:p.n_points
        for r in 1:10
            total += @inbounds(W[r, pp, s, a]) * @inbounds(B[r, qq, s, a])
        end
    end
    @inbounds hess[pair, a] = total + @inbounds(zeta[pair, a])
end

# ------------------------------------------ GPU: per-point stages over threads

@inline load_point_input(Z, s, a) = SVector{10}(ntuple(r -> @inbounds(Z[r, s, a]), Val(10)))

"The throttle as a dual on direction 57 within the chunk c0+1 : c0+C."
@inline throttle_chunk_dual(p::UTParams, x, theta_index, theta_fixed, a, c0, chunk) =
    chunk_seed(TagArc, SVector(arc_throttle(p, x, theta_index, theta_fixed, a)), 56, c0, chunk)[1]

"`arc_duals` seeded on the directions c0+1 : c0+C only."
@inline function arc_chunk_duals(p::UTParams, x, theta_index, theta_fixed, arc_std, a, c0, chunk::Val{C}, full) where {C}
    m, theta, k, L = load_arc(x, theta_index, theta_fixed, a)
    gamma = throttle_chunk_dual(p, x, theta_index, theta_fixed, a, c0, chunk)
    return chunk_seed(TagArc, m, 0, c0, chunk),
        gamma * chunk_seed(TagArc, theta, 7, c0, chunk),
        chunk_seed(TagArc, k, 10, c0, chunk), spread_duals(TagArc, p, start_scale(full, arc_std, a), L, c0, chunk)
end

"Sigma-point inputs z_s and one chunk of the columns of W_s = dz_s/dtheta."
@kernel function ut_point_input_kernel!(Z, W_out, @Const(x), @Const(theta_index), @Const(theta_fixed),
        @Const(arc_std), @Const(gain_scale), @Const(steps), p::UTParams, chunk, full)
    index = @index(Global, Linear)
    C = chunk_width(chunk)
    chunks = cld(ntheta(p), C)
    c, point = (index - 1) % chunks + 1, (index - 1) ÷ chunks
    s, a = point % p.n_points + 1, point ÷ p.n_points + 1
    c0 = C * (c - 1)
    h = @inbounds steps[a]
    md, fd, kd, S = arc_chunk_duals(p, x, theta_index, theta_fixed, arc_std, a, c0, chunk, full)
    K = @inbounds(gain_scale[a]) * SMatrix{3,6}(kd)
    xs, us, _ = point_input(p, md, fd, K, S, s, h)
    for r in 1:10
        zr = r <= 7 ? xs[r] : us[r-7]
        c == 1 && (@inbounds Z[r, s, a] = ForwardDiff.value(zr))
        for j in 1:C
            c0 + j <= ntheta(p) && (@inbounds W_out[r, c0 + j, s, a] = partial(zr, j))
        end
    end
end

"Propagated sigma points y_s and J_s = dPhi/dz."
@kernel function ut_point_step_kernel!(Y, Jpt, @Const(Z), @Const(steps), p::UTParams)
    index = @index(Global, Linear)
    point = index - 1
    s, a = point % p.n_points + 1, point ÷ p.n_points + 1
    h = @inbounds steps[a]
    z = load_point_input(Z, s, a)
    y = sigma_step(p, seed(TagPoint, point_state(z), 0, Val(10)), seed(TagPoint, point_control(z), 7, Val(10)),
        wiener_increment(p, outer_index(p, s), h), h, s)
    for i in 1:7
        @inbounds Y[i, s, a] = ForwardDiff.value(y[i])
        for q in 1:10
            @inbounds Jpt[i, q, s, a] = partial(y[i], q)
        end
    end
end

"Column q of V_s = J_s W_s."
@kernel function ut_point_v_kernel!(V, @Const(Jpt), @Const(W), p::UTParams)
    index = @index(Global, Linear)
    q, point = (index - 1) % ntheta(p) + 1, (index - 1) ÷ ntheta(p)
    s, a = point % p.n_points + 1, point ÷ p.n_points + 1
    column = SVector{10}(ntuple(r -> @inbounds(W[r, q, s, a]), Val(10)))
    v = load_point_jacobian(Jpt, s, a) * column
    for i in 1:7
        @inbounds V[i, q, s, a] = v[i]
    end
end

"Forward-over-reverse step: columns of Hess_z(g_s' Phi), J_s and mu_s = J_s' g_s."
@kernel function ut_point_adjoint_kernel!(Hraw, Jpt, mu_out, @Const(Z), @Const(Y), @Const(ybar), @Const(aN),
        @Const(steps), p::UTParams)
    index = @index(Global, Linear)
    point = index - 1
    s, a = point % p.n_points + 1, point ÷ p.n_points + 1
    h = @inbounds steps[a]
    y, zb = point_adjoint(p, load_point_input(Z, s, a), wiener_increment(p, outer_index(p, s), h), Y, ybar, aN, s, a, h)
    for r in 1:10
        @inbounds mu_out[r, s, a] = ForwardDiff.value(zb[r])
        for q in 1:10
            @inbounds Hraw[r, q, s, a] = partial(zb[r], q)
        end
    end
    for i in 1:7, q in 1:10
        @inbounds Jpt[i, q, s, a] = partial(y[i], q)
    end
end

"H_s, the symmetrized forward-over-reverse Hessian."
@kernel function ut_point_curvature_kernel!(M_out, @Const(Hraw), p::UTParams)
    index = @index(Global, Linear)
    point = index - 1
    s, a = point % p.n_points + 1, point ÷ p.n_points + 1
    for q in 1:10, r in 1:10
        @inbounds M_out[r, q, s, a] = (Hraw[r, q, s, a] + Hraw[q, r, s, a]) / 2
    end
end

"Column q of B_s = H_s W_s and of V_s = J_s W_s."
@kernel function ut_point_bv_kernel!(B_out, V_out, @Const(M_buf), @Const(Jpt), @Const(W), p::UTParams)
    index = @index(Global, Linear)
    q, point = (index - 1) % ntheta(p) + 1, (index - 1) ÷ ntheta(p)
    s, a = point % p.n_points + 1, point ÷ p.n_points + 1
    column = SVector{10}(ntuple(r -> @inbounds(W[r, q, s, a]), Val(10)))
    M = SMatrix{10,10,Float64}(ntuple(l -> @inbounds(M_buf[(l-1)%10+1, (l-1)÷10+1, s, a]), Val(100)))
    b = M * column
    v = load_point_jacobian(Jpt, s, a) * column
    for r in 1:10
        @inbounds B_out[r, q, s, a] = b[r]
    end
    for i in 1:7
        @inbounds V_out[i, q, s, a] = v[i]
    end
end

const ASSEMBLY_TILE = 8                                   # 56 = 7 tiles of 8; 57 = 8 tiles, the last one partial
"Lower-triangle tile pairs of the Hessian assembly."
@inline assembly_tile_pairs(p::UTParams) = (t = cld(ntheta(p), ASSEMBLY_TILE); t * (t + 1) ÷ 2)

"""
Hessian assembly with one work-group per (lower-triangle tile pair, arc): the 10 x 8
slices of W_s and B_s are staged in shared memory once per sigma point and reused by
the 64 threads of the tile, each summing over s, then r.
"""
@kernel function ut_hessian_assemble_tiled_kernel!(hess, @Const(W), @Const(B), @Const(zeta), p::UTParams)
    group = @index(Group, Linear)
    local_index = @index(Local, Linear)
    Wt = @localmem Float64 (10, ASSEMBLY_TILE)
    Bt = @localmem Float64 (10, ASSEMBLY_TILE)
    nt = ntheta(p)
    tile_pair, a = (group - 1) % assembly_tile_pairs(p) + 1, (group - 1) ÷ assembly_tile_pairs(p) + 1
    tp = 1
    while tp * (tp + 1) ÷ 2 < tile_pair
        tp += 1
    end
    tq = tile_pair - tp * (tp - 1) ÷ 2
    lp, lq = (local_index - 1) % ASSEMBLY_TILE + 1, (local_index - 1) ÷ ASSEMBLY_TILE + 1
    pp, qq = ASSEMBLY_TILE * (tp - 1) + lp, ASSEMBLY_TILE * (tq - 1) + lq
    total = 0.0
    for s in 1:p.n_points
        for e in local_index:ASSEMBLY_TILE^2:20ASSEMBLY_TILE
            r, column = (e - 1) % 10 + 1, ((e - 1) ÷ 10) % ASSEMBLY_TILE + 1
            if e <= 10ASSEMBLY_TILE
                q = ASSEMBLY_TILE * (tp - 1) + column
                @inbounds Wt[r, column] = q <= nt ? W[r, q, s, a] : 0.0
            else
                q = ASSEMBLY_TILE * (tq - 1) + column
                @inbounds Bt[r, column] = q <= nt ? B[r, q, s, a] : 0.0
            end
        end
        @synchronize
        for r in 1:10
            total += @inbounds(Wt[r, lp]) * @inbounds(Bt[r, lq])
        end
        @synchronize
    end
    if pp >= qq && pp <= nt
        position = lower_position(pp, qq)
        @inbounds hess[position, a] = total + zeta[position, a]
    end
end

# ---------------------------------------------------------------- callbacks

function ut_values!(out, x, ut::UTOracle)
    d, w, p, n = ut.data, ut.workspace, ut.params, ut.n
    fill!(out, 0.0)   # other oracles write the remaining rows of the same blocks
    launch!(ut_forward_kernel!, n * p.n_points, w.Y, x, d.theta_index, d.theta_fixed, d.arc_std, d.gain_scale,
        d.steps, p, ut.full)
    launch!(ut_mean_kernel!, n * NX, out, w.ybar, w.Y, d.rows, d.ratio, p)
    launch!(ut_deviation_kernel!, n * p.n_points, w.Dn, w.Y, w.ybar, d.inv_next, p, ut.full)
    launch!(ut_covariance_kernel!, n * NL, out, w.Dn, d.rows, d.inv_next, p, ut.full)
    return nothing
end

function launch_point_inputs!(x, ut::UTOracle)
    d, w, p, n = ut.data, ut.workspace, ut.params, ut.n
    launch!(ut_point_input_kernel!, n * p.n_points * cld(ntheta(p), chunk_width(GPU_ARC_CHUNK)), w.Z, w.W, x,
        d.theta_index, d.theta_fixed, d.arc_std, d.gain_scale, d.steps, p, GPU_ARC_CHUNK, ut.full)
end

function ut_jacobian!(vals, x, ut::UTOracle)
    d, w, p, n = ut.data, ut.workspace, ut.params, ut.n
    P = p.n_points
    if ut.gpu
        launch_point_inputs!(x, ut)
        launch!(ut_point_step_kernel!, n * P, w.Y, w.Jpt, w.Z, d.steps, p)
        launch!(ut_point_v_kernel!, n * P * ntheta(p), w.V, w.Jpt, w.W, p)
    else
        launch!(ut_point_jacobian_kernel!, n * P, w.Y, w.V, x, d.theta_index, d.theta_fixed, d.arc_std,
            d.gain_scale, d.steps, p, ut.full)
    end
    launch!(ut_reduce_kernel!, n * NX * (ntheta(p) + 1), w.ybar, w.Vbar, w.Y, w.V, p)
    launch!(ut_normalize_kernel!, n * P * (ntheta(p) + 1), w.V, w.Dn, w.Y, w.ybar, w.Vbar, d.inv_next, p, ut.full)
    launch!(ut_jacobian_kernel!, n * NUT_ROWS * ntheta(p), w.jac, w.Dn, w.V, w.Vbar, d.ratio, d.inv_next, p, ut.full)
    gather!(vals, w.jac, d.jac_gather)
    return nothing
end

function ut_hessian!(vals, x, y, ut::UTOracle)
    d, w, p, n = ut.data, ut.workspace, ut.params, ut.n
    P = p.n_points
    launch!(ut_forward_kernel!, n * P, w.Y, x, d.theta_index, d.theta_fixed, d.arc_std, d.gain_scale, d.steps, p, ut.full)
    launch!(ut_hessian_prep_kernel!, n * NUT_ROWS, w.ybar, w.aN, w.Y, y, d.rows, d.ratio, d.inv_next, p, ut.full)
    launch!(ut_weight_kernel!, n, w.aN, d.inv_next, p, ut.full)
    if ut.gpu
        launch_point_inputs!(x, ut)
        launch!(ut_point_adjoint_kernel!, n * P, w.Hraw, w.Jpt, w.mu, w.Z, w.Y, w.ybar, w.aN, d.steps, p)
        launch!(ut_point_curvature_kernel!, n * P, w.M, w.Hraw, p)
        launch!(ut_point_bv_kernel!, n * P * ntheta(p), w.B, w.V, w.M, w.Jpt, w.W, p)
    else
        launch!(ut_point_hessian_kernel!, n * P, w.W, w.B, w.V, w.Jpt, w.mu, w.Y, w.ybar, w.aN, x, d.theta_index,
            d.theta_fixed, d.arc_std, d.gain_scale, d.steps, p, ut.full)
    end
    launch!(ut_reduce_kernel!, n * NX * (ntheta(p) + 1), w.ybar, w.Vbar, w.Y, w.V, p)   # Vbar
    launch!(ut_gauss_newton_kernel!, n * P * ntheta(p), w.B, w.V, w.Vbar, w.Jpt, w.aN, p)
    launch!(ut_command_curvature_kernel!, n * 2p.d, w.gH, w.mu, x, d.theta_index, d.theta_fixed, d.arc_std,
        d.gain_scale, d.steps, p, ut.full)
    launch!(ut_zeta_kernel!, n * nhess(p), w.zeta, w.gH, x, d.theta_index, d.theta_fixed, d.arc_std,
        d.gain_scale, d.pair_p, d.pair_q, p, ut.full)
    if ut.gpu
        launch!(ut_hessian_assemble_tiled_kernel!, n * assembly_tile_pairs(p) * ASSEMBLY_TILE^2, w.hess, w.W, w.B,
            w.zeta, p; workgroup = ASSEMBLY_TILE^2)
    else
        launch!(ut_hessian_assemble_kernel!, n * nhess(p), w.hess, w.W, w.B, w.zeta, d.pair_p, d.pair_q, p)
    end
    gather!(vals, w.hess, d.hess_gather)
    return nothing
end

# -------------------------------------------------------------------- setup

"""
    ut_oracle(layout, p, backend) -> UTOracle

Index maps, sparsity and workspace of the UT oracle of the arcs in `layout` on
`backend` (`nothing`: CPU), all propagated with the UT parameters `p`. `layout` holds
the 56 x n `theta_index` (local input indices, 0 for fixed data) and `theta_fixed`,
the 35 x n output `rows`, the 7 x n mean-row `ratio`, the start node scales `arc_std` and end
inverse scales `inv_next` (7 x n diagonal, or 28 x n packed lower with `full = Val(true)`), and
the per-arc `gain_scale` and `steps`.
"""
function ut_oracle(layout, p::UTParams, backend)
    (; theta_index, theta_fixed, rows, ratio, arc_std, inv_next, gain_scale, steps) = layout
    n = size(theta_index, 2)
    nt = ntheta(p)
    size(theta_index, 1) == nt || error("the UT layout needs $nt arc inputs")
    pair_p = [pp for pp in 1:nt for _ in 1:pp]
    pair_q = [qq for pp in 1:nt for qq in 1:pp]
    jac_rows, jac_cols, jac_gather = Int[], Int[], Int[]
    hess_rows, hess_cols, hess_gather = Int[], Int[], Int[]
    for a in 1:n
        for row in 1:NUT_ROWS, q in 1:nt
            theta_index[q, a] == 0 && continue
            push!(jac_rows, rows[row, a])
            push!(jac_cols, theta_index[q, a])
            push!(jac_gather, row + NUT_ROWS * (q - 1) + NUT_ROWS * nt * (a - 1))
        end
        for pp in 1:nt, qq in 1:pp
            ip, iq = theta_index[pp, a], theta_index[qq, a]
            (ip == 0 || iq == 0) && continue
            push!(hess_rows, max(ip, iq))
            push!(hess_cols, min(ip, iq))
            push!(hess_gather, lower_position(pp, qq) + nhess(p) * (a - 1))
        end
    end
    device(v) = ExaModels.convert_array(v, backend)
    data = UTOracleData(device(Matrix{Int}(theta_index)), device(Matrix{Float64}(theta_fixed)),
        device(Matrix{Int}(rows)), device(Matrix{Float64}(ratio)), device(Matrix{Float64}(arc_std)),
        device(Matrix{Float64}(inv_next)), device(Vector{Float64}(gain_scale)), device(Vector{Float64}(steps)),
        device(pair_p), device(pair_q), device(jac_gather), device(hess_gather))
    gpu = backend !== nothing
    P, m = p.n_points, gpu ? n : 0
    buffer(dims...) = device(zeros(dims...))
    # aN holds a (7) and N (7 x 7): 56 rows whatever the number of inputs
    workspace = UTWorkspace(buffer(NX, P, n), buffer(NX, P, n), buffer(2NX, n), buffer(10, nt, P, n), buffer(NX, nt, P, n),
        buffer(10, nt, P, n), buffer(10, P, n), buffer(NX, 10, P, n), buffer(2NX, nt, n), buffer(NX + NX^2, n),
        buffer(NUT_ROWS, nt, n), buffer(9, 2p.d, n), buffer(nhess(p), n), buffer(nhess(p), n),
        buffer(10, P, m), buffer(10, 10, P, m), buffer(10, 10, P, m))
    return UTOracle(data, workspace, (; jac_rows, jac_cols, hess_rows, hess_cols), p, n, gpu, get(layout, :full, Val(false)))
end
