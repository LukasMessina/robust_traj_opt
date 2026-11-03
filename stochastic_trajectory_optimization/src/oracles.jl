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

# Square-root covariance defects. The covariance rows of arc k are L_{k+1} - R_k in place of
# L_{k+1} L_{k+1}' - V_k, with R_k = chol(V_k) of the scaled UT covariance (the UT oracle's covariance rows):
# linear in L_{k+1} and, for a linear arc map, homogeneous of degree one in L_k, so a step along the
# constraints is not thrown off them by the dL dL' term.
#
# With mu the multipliers of the R rows and D_V = dR/dV, the Hessian is the exact composition with the
# UT oracle (rows -V, Jacobian -J_V):
#
#   Hess(mu' (-R)) = sum_s (D_V' mu)_s Hess(-V_s) - J_V' Hess_V(mu' chol)(V) J_V,
#
# the first term the UT oracle's with the multipliers D_V' mu. On the CPU the Cholesky terms come from
# ForwardDiff on the host. On a device they are kernels and nothing crosses to the host: the Cholesky factor per arc, and every derivative
# as a directional derivative through the unrolled static Cholesky on dual numbers, one per thread
# (`factor_jacobian_kernel!`: dR along a column of J_V; `factor_multiplier_kernel!`: (D_V' mu)_s;
# `factor_extra_kernel!`: the second derivative of mu' chol along two columns of J_V).

"Packed lower Cholesky factor (LTRI order) of the packed symmetric 7 x 7 matrix `v`; NaN if not positive definite."
function packed_cholesky(v::AbstractVector{T}) where {T}
    L = zeros(T, NX, NX)
    for j in 1:NX
        pivot = v[packed_index(j, j)] - sum(L[j, k]^2 for k in 1:j-1; init = zero(T))
        pivot > 0 || return fill(T(NaN), NL)
        L[j, j] = sqrt(pivot)
        for i in j+1:NX
            L[i, j] = (v[packed_index(i, j)] - sum(L[i, k] * L[j, k] for k in 1:j-1; init = zero(T))) / L[j, j]
        end
    end
    return [L[LTRI_ROWS[r], LTRI_COLS[r]] for r in 1:NL]
end

mutable struct UTFactorOracle{U<:UTOracle,V,M}
    ut::U
    rows::Matrix{Int}          # host copy of the UT oracle's 35 x n output rows
    moments::V                 # UT oracle rows (-mean, -V) at the evaluation point
    jacobian::V                # scratch for the UT oracle's gathered Jacobian
    extra::M                   # 1596 x n  -J_V' Hess_V(mu' chol) J_V
    extra_host::Matrix{Float64}
    gathered::V
    multipliers::V             # y with the multipliers D_V' mu on the covariance rows
end

"The array itself on the CPU, a host copy of a device array."
host(A::Array) = A
host(A) = Array(A)

function ut_factor_oracle(ut::UTOracle)
    w = ut.workspace
    vector(n) = fill!(similar(w.hess, n), 0.0)
    return UTFactorOracle(ut, Matrix{Int}(host(ut.data.rows)), vector(maximum(host(ut.data.rows))),
        vector(length(ut.data.jac_gather)), fill!(similar(w.hess), 0.0), zeros(nhess(ut.params), ut.n),
        vector(length(ut.data.hess_gather)), vector(0))
end

# ------------------------------------------------------------------ device kernels

"Symmetric 7 x 7 matrix of the packed lower entries `v` (any number type)."
@inline symmetric_packed(v::SVector{28}) = SMatrix{7,7}(ntuple(Val(49)) do linear
    r, c = (linear - 1) % 7 + 1, (linear - 1) ÷ 7 + 1
    v[packed_index(max(r, c), min(r, c))]
end)

"Packed lower Cholesky factor of the packed symmetric `v` on a device (NaN where not positive definite)."
@inline device_cholesky(v::SVector{28}) = cholesky_packed(symmetric_packed(v), -Inf)

"Scaled UT covariance V of arc `a` from the oracle rows (-V) in `moments`."
@inline arc_covariance(moments, rows, a) = SVector{28}(ntuple(r -> -@inbounds(moments[rows[NX + r, a]]), Val(28)))

@inline direction_dual(v, d) = ForwardDiff.Dual{TagInner}(v, ForwardDiff.Partials{1,Float64}((d,)))

"Factor rows -R of arc `a` in place of the covariance rows -V."
@kernel function factor_value_kernel!(out, @Const(rows))
    a = @index(Global, Linear)
    R = device_cholesky(arc_covariance(out, rows, a))
    for r in 1:NL
        @inbounds out[rows[NX + r, a]] = -R[r]
    end
end

"Column q of the factor rows' Jacobian, dchol(V)[J_V column q], in place of the covariance rows' (35 x ntheta x n)."
@kernel function factor_jacobian_kernel!(jac, @Const(moments), @Const(rows), nt)
    index = @index(Global, Linear)
    q, a = (index - 1) % nt + 1, (index - 1) ÷ nt + 1
    V = arc_covariance(moments, rows, a)
    R = device_cholesky(SVector{28}(ntuple(r -> direction_dual(V[r], @inbounds(jac[NX + r, q, a])), Val(28))))
    for r in 1:NL
        @inbounds jac[NX + r, q, a] = ForwardDiff.partials(R[r])[1]
    end
end

"Multiplier (D_V' mu)_s of the oracle's covariance row s of arc a, written over the copy of `y`."
@kernel function factor_multiplier_kernel!(multipliers, @Const(y), @Const(moments), @Const(rows))
    index = @index(Global, Linear)
    s, a = (index - 1) % NL + 1, (index - 1) ÷ NL + 1
    V = arc_covariance(moments, rows, a)
    R = device_cholesky(SVector{28}(ntuple(r -> direction_dual(V[r], Float64(r == s)), Val(28))))
    total = 0.0
    for r in 1:NL
        total += @inbounds(y[rows[NX + r, a]]) * ForwardDiff.partials(R[r])[1]
    end
    @inbounds multipliers[rows[NX + s, a]] = total
end

"-(J_V' Hess_V(mu' chol)(V) J_V)[pp, qq]: the second derivative of mu' chol(V) along columns pp and qq of J_V."
@kernel function factor_extra_kernel!(extra, @Const(y), @Const(moments), @Const(rows), @Const(jac), @Const(pair_p),
        @Const(pair_q), nh)
    index = @index(Global, Linear)
    pair, a = (index - 1) % nh + 1, (index - 1) ÷ nh + 1
    pp, qq = @inbounds(pair_p[pair]), @inbounds(pair_q[pair])
    V = arc_covariance(moments, rows, a)
    R = device_cholesky(SVector{28}(ntuple(Val(28)) do r
        inner = direction_dual(V[r], @inbounds(jac[NX + r, qq, a]))
        shift = direction_dual(@inbounds(jac[NX + r, pp, a]), 0.0)
        ForwardDiff.Dual{TagOuter}(inner, ForwardDiff.Partials{1,typeof(inner)}((shift,)))
    end))
    total = 0.0
    for r in 1:NL
        total += @inbounds(y[rows[NX + r, a]]) * pair_value(R[r])
    end
    @inbounds extra[pair, a] = -total
end

# ------------------------------------------------------------------- callbacks

"Scaled UT covariance of arc `a` (packed) from the host copy of the oracle rows."
factor_covariance(f::UTFactorOracle, moments, a) = [-moments[f.rows[NX + r, a]] for r in 1:NL]

function ut_factor_values!(out, x, f::UTFactorOracle)
    ut_values!(out, x, f.ut)
    f.ut.gpu && return launch!(factor_value_kernel!, f.ut.n, out, f.ut.data.rows)
    values, rows = host(out), f.rows
    for a in 1:f.ut.n
        R = packed_cholesky([-values[rows[NX + r, a]] for r in 1:NL])
        for r in 1:NL
            values[rows[NX + r, a]] = -R[r]
        end
    end
    values === out || copyto!(out, values)
    return nothing
end

function ut_factor_jacobian!(vals, x, f::UTFactorOracle)
    ut = f.ut
    ut_values!(f.moments, x, ut)
    ut_jacobian!(vals, x, ut)
    if ut.gpu
        nt = ntheta(ut.params)
        launch!(factor_jacobian_kernel!, nt * ut.n, ut.workspace.jac, f.moments, ut.data.rows, nt)
        return gather!(vals, ut.workspace.jac, ut.data.jac_gather)
    end
    J = host(ut.workspace.jac)
    moments = host(f.moments)
    for a in 1:ut.n
        D = ForwardDiff.jacobian(packed_cholesky, factor_covariance(f, moments, a))
        J[NX+1:NX+NL, :, a] = D * J[NX+1:NX+NL, :, a]
    end
    J === ut.workspace.jac || copyto!(ut.workspace.jac, J)
    gather!(vals, ut.workspace.jac, ut.data.jac_gather)
    return nothing
end

function ut_factor_hessian!(vals, x, y, f::UTFactorOracle)
    ut = f.ut
    rows = f.rows
    ut_values!(f.moments, x, ut)
    ut_jacobian!(f.jacobian, x, ut)   # -J_V
    ut.gpu && return device_factor_hessian!(vals, x, y, f)
    J = host(ut.workspace.jac)
    moments = host(f.moments)
    yh = Array(y)                     # multipliers with D_V' mu on the covariance rows
    for a in 1:ut.n
        v = factor_covariance(f, moments, a)
        mu = [yh[rows[NX + r, a]] for r in 1:NL]
        D = ForwardDiff.jacobian(packed_cholesky, v)
        lambda = D' * mu
        for r in 1:NL
            yh[rows[NX + r, a]] = lambda[r]
        end
        M = ForwardDiff.hessian(u -> dot(mu, packed_cholesky(u)), v)
        Jc = J[NX+1:NX+NL, :, a]
        E = Jc' * M * Jc
        for pp in 1:ntheta(ut.params), qq in 1:pp
            f.extra_host[lower_position(pp, qq), a] = -E[pp, qq]
        end
    end
    length(f.multipliers) == length(y) || (f.multipliers = fill!(similar(ut.workspace.hess, length(y)), 0.0))
    copyto!(f.multipliers, yh)
    ut_hessian!(vals, x, f.multipliers, ut)
    copyto!(f.extra, f.extra_host)
    gather!(f.gathered, f.extra, ut.data.hess_gather)
    vals .+= f.gathered
    return nothing
end

"The Hessian on the device: multipliers D_V' mu and the -J_V' M J_V pairs as kernels, then the UT oracle's Hessian."
function device_factor_hessian!(vals, x, y, f::UTFactorOracle)
    ut = f.ut
    d, n = ut.data, ut.n
    length(f.multipliers) == length(y) || (f.multipliers = fill!(similar(ut.workspace.hess, length(y)), 0.0))
    copyto!(f.multipliers, y)
    launch!(factor_multiplier_kernel!, NL * n, f.multipliers, y, f.moments, d.rows)
    launch!(factor_extra_kernel!, nhess(ut.params) * n, f.extra, y, f.moments, d.rows, ut.workspace.jac, d.pair_p, d.pair_q,
        nhess(ut.params))
    ut_hessian!(vals, x, f.multipliers, ut)
    gather!(f.gathered, f.extra, d.hess_gather)
    vals .+= f.gathered
    return nothing
end

# Thrust chance constraint of step 3, Gamma_k + psi rho(T_k) <= 1, as an ExaModels VectorNonlinearOracle
# (one row per controlled arc). T_k is the closed form of the nested-UT executed-control covariance
# (`control_covariance`), a polynomial of (S = Gamma d, K_hat, L_tilde rows 1..6); rho is the smoothed
# trigonometric largest-eigenvalue formula of the reference. Inputs: the direction d (1..3), K_hat (4..21),
# L_tilde (22..42) and the throttle Gamma (43). Exact derivatives come from single and pair-seeded duals.

const NCHANCE = 43                              # d 3, K_hat 18, L_tilde 21, Gamma
@inline nchance(::UTParams) = NCHANCE
@inline nchance_pairs(p::UTParams) = nchance(p) * (nchance(p) + 1) ÷ 2

struct ChanceData{VI,MI,VF,MF}
    index::MI        # 43 x n global variable index (0: fixed data)
    fixed::MF        # 43 x n values of the fixed inputs
    arc_std::MF      # start node scales: 7 x n standard deviations, or 28 x n packed lower (full)
    gain_scale::VF
    pair_p::VI
    pair_q::VI
    jac_gather::VI
    hess_gather::VI
end

@inline function chance_input(x, index, fixed, q, a)
    i = @inbounds index[q, a]
    return i == 0 ? @inbounds(fixed[q, a]) : @inbounds(x[i])
end

@inline function chance_inputs(x, index, fixed, a)
    f = SVector{3}(ntuple(i -> chance_input(x, index, fixed, i, a), Val(3)))
    k = SVector{18}(ntuple(i -> chance_input(x, index, fixed, 3 + i, a), Val(18)))
    L = SVector{21}(ntuple(i -> chance_input(x, index, fixed, 21 + i, a), Val(21)))
    return f, k, L
end

"Throttle Gamma of a row (input 43) and its single and pair seeds."
@inline chance_throttle(x, index, fixed, a) = chance_input(x, index, fixed, 43, a)
@inline throttle_seed(gamma, q) = single_seed(gamma, q == 43)
@inline throttle_pair_seed(gamma, pp, qq) = pair_seed(gamma, pp == 43, qq == 43)

"Gamma + psi rho(T) with S = Gamma d (T holds Q_G(S) = Gamma^2 Q_G(d))."
@inline function chance_value(p::UTParams, std, gs, theta, k, L, gamma)
    f = gamma * theta
    K, P6 = gs * SMatrix{3,6}(k), position_velocity_covariance(std, L)
    return gamma + control_spread(p, f, K, P6)
end

@kernel function chance_value_kernel!(out, @Const(x), @Const(index), @Const(fixed), @Const(arc_std), @Const(gain_scale),
        p::UTParams, full)
    a = @index(Global, Linear)
    f, k, L = chance_inputs(x, index, fixed, a)
    @inbounds out[a] = chance_value(p, start_scale(full, arc_std, a), gain_scale[a], f, k, L, chance_throttle(x, index, fixed, a))
end

@kernel function chance_jacobian_kernel!(jac, @Const(x), @Const(index), @Const(fixed), @Const(arc_std),
        @Const(gain_scale), p::UTParams, full)
    i = @index(Global, Linear)
    q, a = (i - 1) % nchance(p) + 1, (i - 1) ÷ nchance(p) + 1
    f, k, L = chance_inputs(x, index, fixed, a)
    fd = SVector{3}(ntuple(j -> single_seed(f[j], q == j), Val(3)))
    kd = SVector{18}(ntuple(j -> single_seed(k[j], q == 3 + j), Val(18)))
    Ld = SVector{21}(ntuple(j -> single_seed(L[j], q == 21 + j), Val(21)))
    gd = throttle_seed(chance_throttle(x, index, fixed, a), q)
    value = chance_value(p, start_scale(full, arc_std, a), @inbounds(gain_scale[a]), fd, kd, Ld, gd)
    @inbounds jac[q, a] = ForwardDiff.partials(value)[1]
end

@kernel function chance_hessian_kernel!(hess, @Const(x), @Const(y), @Const(index), @Const(fixed), @Const(arc_std),
        @Const(gain_scale), @Const(pair_p), @Const(pair_q), p::UTParams, full)
    i = @index(Global, Linear)
    pair, a = (i - 1) % nchance_pairs(p) + 1, (i - 1) ÷ nchance_pairs(p) + 1
    pp, qq = @inbounds(pair_p[pair]), @inbounds(pair_q[pair])
    f, k, L = chance_inputs(x, index, fixed, a)
    fd = SVector{3}(ntuple(j -> pair_seed(f[j], pp == j, qq == j), Val(3)))
    kd = SVector{18}(ntuple(j -> pair_seed(k[j], pp == 3 + j, qq == 3 + j), Val(18)))
    Ld = SVector{21}(ntuple(j -> pair_seed(L[j], pp == 21 + j, qq == 21 + j), Val(21)))
    gd = throttle_pair_seed(chance_throttle(x, index, fixed, a), pp, qq)
    value = chance_value(p, start_scale(full, arc_std, a), @inbounds(gain_scale[a]), fd, kd, Ld, gd)
    @inbounds hess[pair, a] = y[a] * pair_value(value)
end

"""
Chance-row layout of the controlled arcs with control slots `slots` and start nodes
`start_nodes`: the global indices of the 43 inputs of the row (d, K_hat, L_tilde rows 1..6, Gamma),
0 and the value in `fixed` for data (the fixed initial factor), and the scales of each start node.
"""
function chance_layout(slots, start_nodes, offsets, scaling::CovarianceScaling, gain_scale)
    n = length(slots)
    index = zeros(Int, NCHANCE, n)
    fixed = zeros(size(index))
    for (a, (slot, node)) in enumerate(zip(slots, start_nodes))
        index[1:3, a] = offsets.ff .+ 3(slot-1) .+ (1:3)
        index[end, a] = offsets.throttle + slot
        index[4:21, a] = offsets.gains .+ 18(slot-1) .+ (1:18)
        if node == 1
            fixed[22:42, a] = scaling.initial_factor[1:21]
        else
            index[22:42, a] = offsets.chol .+ 28(node-2) .+ (1:21)
        end
    end
    arc_std = scaling.full ? packed_node_scales(scaling, collect(start_nodes)) : scaling.std[:, collect(start_nodes)]
    return (; index, fixed, arc_std, gain_scale = collect(gain_scale), full = Val(scaling.full))
end

"The thrust chance rows Gamma_k + psi rho(T_k) <= 1 of the arcs in `layout`."
function chance_oracle(layout, nvar::Int, p::UTParams, backend)
    (; index, fixed, arc_std, gain_scale) = layout
    full = get(layout, :full, Val(false))
    n = size(index, 2)
    nq = nchance(p)
    size(index, 1) == nq || error("the chance layout needs $nq inputs")
    pair_p = [pp for pp in 1:nq for _ in 1:pp]
    pair_q = [qq for pp in 1:nq for qq in 1:pp]
    jac_rows, jac_cols, jac_gather = Int[], Int[], Int[]
    hess_rows, hess_cols, hess_gather = Int[], Int[], Int[]
    for a in 1:n
        for q in 1:nq
            index[q, a] == 0 && continue
            push!(jac_rows, a); push!(jac_cols, index[q, a]); push!(jac_gather, q + nq * (a - 1))
        end
        for pp in 1:nq, qq in 1:pp
            ip, iq = index[pp, a], index[qq, a]
            (ip == 0 || iq == 0) && continue
            push!(hess_rows, max(ip, iq)); push!(hess_cols, min(ip, iq))
            push!(hess_gather, lower_position(pp, qq) + nchance_pairs(p) * (a - 1))
        end
    end
    device(v) = ExaModels.convert_array(v, backend)
    d = ChanceData(device(index), device(fixed), device(Matrix{Float64}(arc_std)), device(Vector{Float64}(gain_scale)),
        device(pair_p), device(pair_q), device(jac_gather), device(hess_gather))
    jac_buffer = device(zeros(nq, n))
    hess_buffer = device(zeros(nchance_pairs(p), n))
    return ExaModels.VectorNonlinearOracle(; nvar, ncon=n,
        jac_rows, jac_cols, hess_rows, hess_cols,
        lcon=device(fill(-Inf, n)), ucon=device(ones(n)),
        f! = (out, x) -> launch!(chance_value_kernel!, n, out, x, d.index, d.fixed, d.arc_std, d.gain_scale, p, full),
        jac! = function (vals, x)
            launch!(chance_jacobian_kernel!, n * nq, jac_buffer, x, d.index, d.fixed, d.arc_std, d.gain_scale, p, full)
            gather!(vals, jac_buffer, d.jac_gather)
        end,
        hess! = function (vals, x, y)
            launch!(chance_hessian_kernel!, n * nchance_pairs(p), hess_buffer, x, y, d.index, d.fixed, d.arc_std,
                d.gain_scale, d.pair_p, d.pair_q, p, full)
            gather!(vals, hess_buffer, d.hess_gather)
        end)
end

# Cholesky-slack (epigraph) form of the thrust chance constraint of step 4: per controlled arc k the 6 packed
# lower rows
#
#   tau_hat_k^2 I - G_hat_k G_hat_k' - T_k / s_k^2 = 0,
#
# with T_k the closed form `control_covariance` of the chance rows (the inputs of `chance_layout`: d, K_hat,
# L_tilde rows 1..6 and the throttle) and s_k = sqrt(lambda_max(T_k)) at the start, a fixed scale. G_hat_k is
# lower triangular with diagonal >= CHANCE_SLACK_FLOOR / s_k, so tau_k = s_k tau_hat_k >= rho(T_k) without an
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
@inline function slack_values(p::UTParams, std, gs, theta, k, L, gamma, tau, g, kappa)
    f = gamma * theta
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
    v = slack_values(p, start_scale(full, arc_std, a), @inbounds(gain_scale[a]), f, k, L, chance_throttle(x, index, fixed, a), tau, g,
        @inbounds(kappa[a]))
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
    gd = throttle_seed(chance_throttle(x, index, fixed, a), q)
    td = single_seed(tau, q == nq + 1)
    Gd = SVector{NSLACK}(ntuple(j -> single_seed(g[j], q == nq + 1 + j), Val(NSLACK)))
    v = slack_values(p, start_scale(full, arc_std, a), @inbounds(gain_scale[a]), fd, kd, Ld,
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
    gd = throttle_pair_seed(chance_throttle(x, index, fixed, a), pp, qq)
    td = pair_seed(tau, pp == nq + 1, qq == nq + 1)
    Gd = SVector{NSLACK}(ntuple(j -> pair_seed(g[j], pp == nq + 1 + j, qq == nq + 1 + j), Val(NSLACK)))
    v = slack_values(p, start_scale(full, arc_std, a), @inbounds(gain_scale[a]), fd, kd, Ld,
        gd, td, Gd, @inbounds(kappa[a]))
    total = 0.0
    for r in 1:NSLACK
        total += @inbounds(y[NSLACK * (a - 1) + r]) * pair_value(v[r])
    end
    @inbounds hess[pair, a] = total
end

"""
The slack rows of the arcs in the chance `layout` (`chance_layout`):
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

"Start of the slack variables: s = sqrt(lambda_max(T)), tau_hat = tau / s and G_hat = chol(tau^2 I - T) / s packed."
function slack_oracle_start(T::AbstractMatrix, floor)
    s, tau_hat, _ = slack_chance_start(T, floor)
    G = cholesky(Symmetric((tau_hat * s)^2 * I - T)).L / s
    return s, tau_hat, [G[SLACK_ROWS[r], SLACK_COLS[r]] for r in 1:NSLACK]
end
