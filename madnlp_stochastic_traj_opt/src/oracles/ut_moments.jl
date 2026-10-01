# UT moment oracle: the propagated node mean and normalized covariance of every
# arc, added (with a minus sign) to the native rows `means[:, k+1]` and
# `pack(L_{k+1} L_{k+1}')` through ExaModels.add_eval. No decision variable is
# introduced.
#
# Arc inputs theta = (m 7, f 3, K_hat 18, L_tilde 28); the first arc starts at the
# fixed initial factor, so its theta has 28 active entries. For sigma point s,
# z_s = (x_s, u_s) = Z_s(theta) and y_s = Phi(z_s) is one XMDS2 RK9 step. With
# l = a'ybar + tr(B C) and g_s = dl/dy_s = w (a + 2 N (y_s - ybar)):
#
#   Hessian = sum_s W_s' (H_s + 2 w J_s' N J_s) W_s - 2 Vbar' N Vbar + Hess(zeta)
#
# W_s = dZ_s/dtheta, J_s = dPhi/dz, H_s = Hess_z(g_s' Phi) (forward-over-reverse),
# V_s = J_s W_s, Vbar = sum_s w V_s, zeta(theta) = sum_s (J_s' g_s)' Z_s(theta).
# Every term is exact; the last one carries the curvature of the spread Cholesky
# factor, of the bilinear gain map and of the Gates factors.
#
# The per-arc kernels are shared by both backends. The per-sigma-point derivatives
# (W_s, J_s, H_s and the products V_s, B_s) are organized per backend:
#  * CPU: one thread per sigma point, with dual numbers over all 56 arc inputs;
#  * GPU: stages split over threads to keep the dual numbers small: W_s in chunks of
#    8 of the 56 directions, J_s and H_s per point, V_s and B_s one column per thread,
#    and sum_s W_s' B_s in shared-memory tiles.
# On the same device the two layouts give bitwise identical results: every partial
# derivative comes from the same operations at any chunk width, and the assembly sums
# over s outer and r inner in both. The 10 sigma-point directions are always seeded
# together: StaticArrays' `dot` sums with `@simd`, so its order depends on the dual width.
# Every kernel reads `@index(Global, Linear)` in a plain assignment: KernelAbstractions'
# CPU backend does not rewrite it inside a larger expression.

const NTHETA = 56
const NUT_ROWS = NX + NL                       # 35 oracle rows per arc
const NHESS_ARC = NTHETA * (NTHETA + 1) ÷ 2    # 1596

# GPU: dual-number directions per thread, of the 28 factor and 56 arc-input directions.
const GPU_FACTOR_CHUNK = Val(4)
const GPU_ARC_CHUNK = Val(8)

struct UTOracleData{VI,MI,VF,MF}
    theta_index::MI      # 56 x N local-x index of each arc input (0: fixed data)
    arc_std::MF          # 7 x N node scales of the arc start
    inv_next::MF         # 7 x N inverse node scales of the arc end
    gain_scale::VF
    steps::VF
    L0::VF               # 28 packed initial factor (scaled)
    pair_p::VI           # 1596 dense lower-triangle pairs
    pair_q::VI
    jac_gather::VI
    hess_gather::VI
end

struct UTWorkspace{A2,A3,A4}
    S::A2        # 28 x N spread factors
    dS::A3       # 28 x 28 x N  d spread / d L_tilde
    Y::A3        # 7 x P x N propagated sigma points
    ybar::A2     # 7 x N
    W::A4        # 10 x 56 x P x N  W_s = dz_s/dtheta
    V::A4        # 7 x 56 x P x N  V_s = J_s W_s
    B::A4        # 10 x 56 x P x N  B_s = (H_s + 2 w J_s' N J_s) W_s
    mu::A3       # 10 x P x N  J_s' g_s
    Vbar::A3     # 7 x 56 x N
    aN::A2       # 56 x N: a (7) and N (7x7, column-major)
    jac::A3      # 35 x 56 x N
    zeta::A2     # 1596 x N
    hess::A2     # 1596 x N
    # GPU stages only (empty on the CPU)
    Z::A3        # 10 x P x N sigma-point inputs z_s
    Jpt::A4      # 7 x 10 x P x N  J_s
    Hraw::A4     # 10 x 10 x P x N  forward-over-reverse Hessian columns
    M::A4        # 10 x 10 x P x N  H_s + 2 w J_s' N J_s
end

struct UTOracle{D<:UTOracleData,W<:UTWorkspace,S,FC}
    data::D
    workspace::W
    structure::S         # jac_rows, jac_cols, hess_rows, hess_cols (local-x indices)
    factor_chunk::FC     # Val: factor directions per thread
    gpu::Bool
end

# ------------------------------------------------------------ sigma points

@inline function load_arc(x, theta_index, L0, a)
    m = SVector{7}(ntuple(i -> @inbounds(x[theta_index[i, a]]), Val(7)))
    f = SVector{3}(ntuple(i -> @inbounds(x[theta_index[7+i, a]]), Val(3)))
    k = SVector{18}(ntuple(i -> @inbounds(x[theta_index[10+i, a]]), Val(18)))
    L = a == 1 ? SVector{28}(ntuple(i -> @inbounds(L0[i]), Val(28))) :
                 SVector{28}(ntuple(i -> @inbounds(x[theta_index[28+i, a]]), Val(28)))
    return m, f, k, L
end

@inline arc_column(A, a) = SVector{7}(ntuple(i -> @inbounds(A[i, a]), Val(7)))

"Node covariance P = (std .* L)(std .* L)' and its spread chol(d P + jitter I, jitter)."
@inline function spread_factor(p::UTParams, std::SVector{7}, L::SVector{28})
    F = std .* lower_matrix(L)
    return arc_spread(p, F * F')
end

"Outer sigma point `o` (1..2d): state, commanded control and Wiener increment."
@inline function outer_point(p::UTParams, m::SVector{7}, f::SVector{3}, K::SMatrix{3,6}, S::SVector{28}, o::Int, h)
    TS = eltype(S)
    j = (o - 1) % p.d + 1
    sigma = o <= p.d ? 1.0 : -1.0
    column = j <= NX ? packed_column(S, j) : zero(SVector{7,TS})
    nav = j - NX
    measurement = j <= NX ? SVector{6,TS}(ntuple(i -> column[i], Val(6))) :
        SVector{6,TS}(ntuple(i -> i == nav ? TS(p.nav_root[i]) : zero(TS), Val(6)))
    x = m + sigma * (p.scale .* column)
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

"Sigma point `s` = (outer point, inner Gates point): state, executed control, Wiener increment."
@inline function point_input(p::UTParams, m, f, K, S, s::Int, h)
    o, l = (s - 1) ÷ 6 + 1, (s - 1) % 6 + 1
    x, c, dw = outer_point(p, m, f, K, S, o, h)
    return x, inner_control(p, c, gates_factor(c, p), l), dw
end

@inline sigma_step(p::UTParams, x, u, dw, h) =
    rk_step(XMDS2RK9, x, u, sigma_mass_flow(p, u), diffusion_increment(p.sigma_nd, dw), h, p.mu, p.ve)

@inline point_state(z) = SVector{7}(ntuple(i -> z[i], Val(7)))
@inline point_control(z) = SVector{3}(ntuple(i -> z[7+i], Val(3)))
@inline point_jacobian(y) = SMatrix{7,10,Float64}(ntuple(l -> partial(y[(l-1)%7+1], (l-1)÷7+1), Val(70)))
@inline load_N(aN, a) = SMatrix{7,7,Float64}(ntuple(i -> @inbounds(aN[7+i, a]), Val(49)))

"Weighted mean of component `i` over the sigma points."
@inline function sigma_mean(Y, i, a, p::UTParams)
    total = 0.0
    for s in 1:p.n_points
        total += p.w_point * @inbounds(Y[i, s, a])
    end
    return total
end

"""
Forward-over-reverse RK9 step of sigma point `s` from its input `z`: the propagated
point `y` and `zb = J_s' g_s`, both as duals over the 10 point inputs.
"""
@inline function point_adjoint(p::UTParams, z, dw, Y, ybar, aN, s, a, h)
    N = load_N(aN, a)
    av = SVector{7}(ntuple(i -> @inbounds(aN[i, a]), Val(7)))
    deviation = SVector{7}(ntuple(i -> @inbounds(Y[i, s, a]) - @inbounds(ybar[i, a]), Val(7)))
    g = p.w_point * (av + 2 * (N * deviation))
    xd, ud = seed(TagPoint, point_state(z), 0, Val(10)), seed(TagPoint, point_control(z), 7, Val(10))
    magnitude = sigma_mass_flow(p, ud)
    gdw = diffusion_increment(p.sigma_nd, dw)
    y, stages = rk_step_stages(XMDS2RK9, xd, ud, magnitude, gdw, h, p.mu, p.ve)
    xb, ub, mb = rk_adjoint(XMDS2RK9, stages, ud, magnitude, g, h, p.mu, p.ve)
    return y, vcat(xb, ub + mb * (ud / magnitude))
end

"H_s + 2 w J_s' N J_s, with H_s the symmetrized forward-over-reverse Hessian."
@inline point_curvature(p::UTParams, H, J, N) = H + (2 * p.w_point) * (J' * N * J)

# ------------------------------------------------------------------- values

@kernel function ut_forward_kernel!(Y, @Const(x), @Const(theta_index), @Const(L0), @Const(arc_std),
        @Const(gain_scale), @Const(steps), p::UTParams)
    index = @index(Global, Linear)
    a, s = (index - 1) ÷ p.n_points + 1, (index - 1) % p.n_points + 1
    m, f, k, L = load_arc(x, theta_index, L0, a)
    h = @inbounds steps[a]
    S = spread_factor(p, arc_column(arc_std, a), L)
    K = @inbounds(gain_scale[a]) * SMatrix{3,6}(k)
    xs, us, dw = point_input(p, m, f, K, S, s, h)
    y = sigma_step(p, xs, us, dw, h)
    for i in 1:7
        @inbounds Y[i, s, a] = y[i]
    end
end

"Oracle rows: the mean (rows 1..7, also stored in ybar) and the scaled covariance entries."
@kernel function ut_moments_kernel!(out, ybar, @Const(Y), @Const(inv_next), n::Int, p::UTParams)
    index = @index(Global, Linear)
    a, row = (index - 1) ÷ NUT_ROWS + 1, (index - 1) % NUT_ROWS + 1
    if row <= NX
        m = sigma_mean(Y, row, a, p)
        @inbounds ybar[row, a] = m
        @inbounds out[row + 7(a-1)] = -m
    else
        r = row - NX
        i, j = LTRI_ROWS[r], LTRI_COLS[r]
        mi, mj = sigma_mean(Y, i, a, p), sigma_mean(Y, j, a, p)
        total = 0.0
        for s in 1:p.n_points
            di = (@inbounds(Y[i, s, a]) - mi) / p.scale[i]
            dj = (@inbounds(Y[j, s, a]) - mj) / p.scale[j]
            total += (p.w_point * di) * dj
        end
        @inbounds out[7n + r + 28(a-1)] = -(total * inv_next[i, a] * inv_next[j, a])
    end
end

# ------------------------------------------------- per-arc derivative kernels

"Spread factor and d spread / d L_tilde on one chunk of the 28 factor directions."
@kernel function ut_factor_kernel!(S_out, dS_out, @Const(x), @Const(theta_index), @Const(L0), @Const(arc_std),
        p::UTParams, chunk)
    index = @index(Global, Linear)
    C = chunk_width(chunk)
    chunks = NL ÷ C
    c, a = (index - 1) % chunks + 1, (index - 1) ÷ chunks + 1
    c0 = C * (c - 1)
    _, _, _, L = load_arc(x, theta_index, L0, a)
    std = arc_column(arc_std, a)
    if a == 1   # the first arc starts at the fixed initial factor
        if c == 1
            S = spread_factor(p, std, L)
            for r in 1:NL
                @inbounds S_out[r, a] = S[r]
            end
        end
        for r in 1:NL, j in 1:C
            @inbounds dS_out[r, c0 + j, a] = 0.0
        end
    else
        S = spread_factor(p, std, chunk_seed(TagFactor, L, 0, c0, chunk))
        for r in 1:NL
            c == 1 && (@inbounds S_out[r, a] = ForwardDiff.value(S[r]))
            for j in 1:C
                @inbounds dS_out[r, c0 + j, a] = partial(S[r], j)
            end
        end
    end
end

"Sigma-point mean (q = 0) and Vbar = sum_s w V_s (q = 1..56)."
@kernel function ut_reduce_kernel!(ybar, Vbar, @Const(Y), @Const(V), p::UTParams)
    index = @index(Global, Linear)
    i = (index - 1) % 7 + 1
    q = ((index - 1) ÷ 7) % (NTHETA + 1)
    a = (index - 1) ÷ (7 * (NTHETA + 1)) + 1
    if q == 0
        @inbounds ybar[i, a] = sigma_mean(Y, i, a, p)
    else
        total = 0.0
        for s in 1:p.n_points
            total += p.w_point * @inbounds(V[i, q, s, a])
        end
        @inbounds Vbar[i, q, a] = total
    end
end

@kernel function ut_jacobian_kernel!(jac, @Const(Y), @Const(ybar), @Const(V), @Const(Vbar), @Const(inv_next), p::UTParams)
    index = @index(Global, Linear)
    row = (index - 1) % NUT_ROWS + 1
    q = ((index - 1) ÷ NUT_ROWS) % NTHETA + 1
    a = (index - 1) ÷ (NUT_ROWS * NTHETA) + 1
    if row <= NX
        @inbounds jac[row, q, a] = -Vbar[row, q, a]
    else
        r = row - NX
        i, j = LTRI_ROWS[r], LTRI_COLS[r]
        total = 0.0
        for s in 1:p.n_points
            di = (@inbounds(Y[i, s, a]) - @inbounds(ybar[i, a])) / p.scale[i]
            dj = (@inbounds(Y[j, s, a]) - @inbounds(ybar[j, a])) / p.scale[j]
            ddi = (@inbounds(V[i, q, s, a]) - @inbounds(Vbar[i, q, a])) / p.scale[i]
            ddj = (@inbounds(V[j, q, s, a]) - @inbounds(Vbar[j, q, a])) / p.scale[j]
            total += p.w_point * (ddi * dj + di * ddj)
        end
        @inbounds jac[row, q, a] = -(total * inv_next[i, a] * inv_next[j, a])
    end
end

"""
Sigma-point mean and the multiplier blocks: a = -lambda_mean and
N = diag(1/scale) B diag(1/scale) with tr(B C) = -sum lambda_r inv_i inv_j C_ij.
"""
@kernel function ut_hessian_prep_kernel!(ybar, aN, @Const(Y), @Const(y), @Const(inv_next), n::Int, p::UTParams)
    index = @index(Global, Linear)
    k, a = (index - 1) % (NX + NL) + 1, (index - 1) ÷ (NX + NL) + 1
    if k <= NX
        @inbounds ybar[k, a] = sigma_mean(Y, k, a, p)
        @inbounds aN[k, a] = -y[k + 7(a-1)]
    else
        r = k - NX
        i, j = LTRI_ROWS[r], LTRI_COLS[r]
        lambda = @inbounds y[7n + r + 28(a-1)]
        value = -lambda * inv_next[i, a] * inv_next[j, a] / (p.scale[i] * p.scale[j])
        value = i == j ? value : value / 2
        @inbounds aN[7 + i + 7(j-1), a] = value
        @inbounds aN[7 + j + 7(i-1), a] = value
    end
end

"Second derivative of zeta(theta) = sum_s mu_s' Z_s(theta) along one input pair."
@kernel function ut_zeta_kernel!(zeta, @Const(mu), @Const(x), @Const(theta_index), @Const(L0), @Const(arc_std),
        @Const(gain_scale), @Const(steps), @Const(pair_p), @Const(pair_q), p::UTParams)
    index = @index(Global, Linear)
    pair = (index - 1) % NHESS_ARC + 1
    a = (index - 1) ÷ NHESS_ARC + 1
    pp, qq = @inbounds(pair_p[pair]), @inbounds(pair_q[pair])
    if pp <= 7 || qq <= 7 || (a == 1 && (pp > 28 || qq > 28))
        # zeta is affine in the means; the first arc's factor is data.
        @inbounds zeta[pair, a] = 0.0
    else
        m, f, k, L = load_arc(x, theta_index, L0, a)
        h = @inbounds steps[a]
        fd = SVector{3}(ntuple(i -> pair_seed(f[i], pp == 7 + i, qq == 7 + i), Val(3)))
        kd = SVector{18}(ntuple(i -> pair_seed(k[i], pp == 10 + i, qq == 10 + i), Val(18)))
        Ld = SVector{28}(ntuple(i -> pair_seed(L[i], pp == 28 + i, qq == 28 + i), Val(28)))
        S = spread_factor(p, arc_column(arc_std, a), Ld)
        K = @inbounds(gain_scale[a]) * SMatrix{3,6}(kd)
        total = zero(eltype(S))
        for o in 1:2p.d
            xs, c, _ = outer_point(p, m, fd, K, S, o, h)
            G = gates_factor(c, p)
            for l in 1:6
                s = 6(o - 1) + l
                us = inner_control(p, c, G, l)
                for i in 1:7
                    total += @inbounds(mu[i, s, a]) * xs[i]
                end
                for i in 1:3
                    total += @inbounds(mu[7+i, s, a]) * us[i]
                end
            end
        end
        @inbounds zeta[pair, a] = pair_value(total)
    end
end

"Hessian entry from the point sum `total` = sum_s (W_s' B_s)[pp, qq]: total - 2 Vbar' N Vbar + zeta."
@inline function hessian_entry(total, Vbar, aN, zeta, pp, qq, a)
    quadratic = 0.0
    for j in 1:7, i in 1:7
        quadratic += @inbounds(Vbar[i, pp, a]) * @inbounds(aN[7 + i + 7(j-1), a]) * @inbounds(Vbar[j, qq, a])
    end
    return total - 2quadratic + @inbounds(zeta[lower_position(pp, qq), a])
end

# ------------------------------------------------ CPU: one thread per point

"Arc inputs as 56-direction duals; the spread carries its stored Jacobian."
@inline function arc_duals(x, theta_index, L0, S_buf, dS_buf, a)
    m, f, k, _ = load_arc(x, theta_index, L0, a)
    md = seed(TagArc, m, 0, Val(NTHETA))
    fd = seed(TagArc, f, 7, Val(NTHETA))
    kd = seed(TagArc, k, 10, Val(NTHETA))
    Sd = SVector{28}(ntuple(Val(28)) do r
        ForwardDiff.Dual{TagArc}(@inbounds(S_buf[r, a]), ForwardDiff.Partials{NTHETA,Float64}(
            ntuple(q -> q <= 28 ? 0.0 : @inbounds(dS_buf[r, q-28, a]), Val(NTHETA))))
    end)
    return md, fd, kd, Sd
end

"Sigma-point input z_s, W_s = dz_s/dtheta over all 56 directions, and the Wiener increment."
@inline function point_input_jacobian(p::UTParams, x, theta_index, L0, S_buf, dS_buf, gain_scale, a, s, h)
    md, fd, kd, Sd = arc_duals(x, theta_index, L0, S_buf, dS_buf, a)
    K = @inbounds(gain_scale[a]) * SMatrix{3,6}(kd)
    xs, us, dw = point_input(p, md, fd, K, Sd, s, h)
    z = SVector{10}(ntuple(i -> ForwardDiff.value(i <= 7 ? xs[i] : us[i-7]), Val(10)))
    W = SMatrix{10,NTHETA,Float64}(ntuple(Val(10 * NTHETA)) do linear
        r, q = (linear - 1) % 10 + 1, (linear - 1) ÷ 10 + 1
        partial(r <= 7 ? xs[r] : us[r-7], q)
    end)
    return z, W, dw
end

"Propagated sigma points y_s and V_s = J_s W_s."
@kernel function ut_point_jacobian_kernel!(Y, V, @Const(x), @Const(theta_index), @Const(L0), @Const(S_buf),
        @Const(dS_buf), @Const(gain_scale), @Const(steps), p::UTParams)
    index = @index(Global, Linear)
    a, s = (index - 1) ÷ p.n_points + 1, (index - 1) % p.n_points + 1
    h = @inbounds steps[a]
    z, W, dw = point_input_jacobian(p, x, theta_index, L0, S_buf, dS_buf, gain_scale, a, s, h)
    y = sigma_step(p, seed(TagPoint, point_state(z), 0, Val(10)), seed(TagPoint, point_control(z), 7, Val(10)), dw, h)
    VJ = point_jacobian(y) * W
    for i in 1:7
        @inbounds Y[i, s, a] = ForwardDiff.value(y[i])
        for q in 1:NTHETA
            @inbounds V[i, q, s, a] = VJ[i, q]
        end
    end
end

"W_s, B_s = (H_s + 2 w J_s' N J_s) W_s, V_s = J_s W_s and mu_s = J_s' g_s."
@kernel function ut_point_hessian_kernel!(W_out, B_out, V_out, mu_out, @Const(Y), @Const(ybar), @Const(aN),
        @Const(x), @Const(theta_index), @Const(L0), @Const(S_buf), @Const(dS_buf), @Const(gain_scale),
        @Const(steps), p::UTParams)
    index = @index(Global, Linear)
    a, s = (index - 1) ÷ p.n_points + 1, (index - 1) % p.n_points + 1
    h = @inbounds steps[a]
    z, W, dw = point_input_jacobian(p, x, theta_index, L0, S_buf, dS_buf, gain_scale, a, s, h)
    y, zb = point_adjoint(p, z, dw, Y, ybar, aN, s, a, h)
    J = point_jacobian(y)
    H = SMatrix{10,10,Float64}(ntuple(Val(100)) do linear
        r, q = (linear - 1) % 10 + 1, (linear - 1) ÷ 10 + 1
        (partial(zb[r], q) + partial(zb[q], r)) / 2
    end)
    BW = point_curvature(p, H, J, load_N(aN, a)) * W
    VJ = J * W
    for q in 1:NTHETA
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
    end
end

"Hessian assembly, one lower-triangle entry per thread."
@kernel function ut_hessian_assemble_kernel!(hess, @Const(W), @Const(B), @Const(Vbar), @Const(aN), @Const(zeta),
        @Const(pair_p), @Const(pair_q), p::UTParams)
    index = @index(Global, Linear)
    pair = (index - 1) % NHESS_ARC + 1
    a = (index - 1) ÷ NHESS_ARC + 1
    pp, qq = @inbounds(pair_p[pair]), @inbounds(pair_q[pair])
    total = 0.0
    for s in 1:p.n_points
        for r in 1:10
            total += @inbounds(W[r, pp, s, a]) * @inbounds(B[r, qq, s, a])
        end
    end
    @inbounds hess[pair, a] = hessian_entry(total, Vbar, aN, zeta, pp, qq, a)
end

# ------------------------------------------ GPU: per-point stages over threads

@inline load_point_input(Z, s, a) = SVector{10}(ntuple(r -> @inbounds(Z[r, s, a]), Val(10)))
@inline load_point_jacobian(Jpt, s, a) =
    SMatrix{7,10,Float64}(ntuple(l -> @inbounds(Jpt[(l-1)%7+1, (l-1)÷7+1, s, a]), Val(70)))

"`arc_duals` seeded on the directions c0+1 : c0+C only."
@inline function arc_chunk_duals(x, theta_index, L0, S_buf, dS_buf, a, c0, chunk::Val{C}) where {C}
    m, f, k, _ = load_arc(x, theta_index, L0, a)
    md = chunk_seed(TagArc, m, 0, c0, chunk)
    fd = chunk_seed(TagArc, f, 7, c0, chunk)
    kd = chunk_seed(TagArc, k, 10, c0, chunk)
    Sd = SVector{28}(ntuple(Val(28)) do r
        ForwardDiff.Dual{TagArc}(@inbounds(S_buf[r, a]), ForwardDiff.Partials{C,Float64}(
            ntuple(j -> c0 + j <= 28 ? 0.0 : @inbounds(dS_buf[r, c0 + j - 28, a]), Val(C))))
    end)
    return md, fd, kd, Sd
end

"Sigma-point inputs z_s and one chunk of the columns of W_s = dz_s/dtheta."
@kernel function ut_point_input_kernel!(Z, W_out, @Const(x), @Const(theta_index), @Const(L0), @Const(S_buf),
        @Const(dS_buf), @Const(gain_scale), @Const(steps), p::UTParams, chunk)
    index = @index(Global, Linear)
    C = chunk_width(chunk)
    chunks = NTHETA ÷ C
    c, point = (index - 1) % chunks + 1, (index - 1) ÷ chunks
    s, a = point % p.n_points + 1, point ÷ p.n_points + 1
    c0 = C * (c - 1)
    h = @inbounds steps[a]
    md, fd, kd, Sd = arc_chunk_duals(x, theta_index, L0, S_buf, dS_buf, a, c0, chunk)
    K = @inbounds(gain_scale[a]) * SMatrix{3,6}(kd)
    xs, us, _ = point_input(p, md, fd, K, Sd, s, h)
    for r in 1:10
        zr = r <= 7 ? xs[r] : us[r-7]
        c == 1 && (@inbounds Z[r, s, a] = ForwardDiff.value(zr))
        for j in 1:C
            @inbounds W_out[r, c0 + j, s, a] = partial(zr, j)
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
        wiener_increment(p, (s - 1) ÷ 6 + 1, h), h)
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
    q, point = (index - 1) % NTHETA + 1, (index - 1) ÷ NTHETA
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
    y, zb = point_adjoint(p, load_point_input(Z, s, a), wiener_increment(p, (s - 1) ÷ 6 + 1, h), Y, ybar, aN, s, a, h)
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

"M_s = H_s + 2 w J_s' N J_s, with H_s the symmetrized forward-over-reverse Hessian."
@kernel function ut_point_curvature_kernel!(M_out, @Const(Hraw), @Const(Jpt), @Const(aN), p::UTParams)
    index = @index(Global, Linear)
    point = index - 1
    s, a = point % p.n_points + 1, point ÷ p.n_points + 1
    H = SMatrix{10,10,Float64}(ntuple(Val(100)) do linear
        r, q = (linear - 1) % 10 + 1, (linear - 1) ÷ 10 + 1
        (@inbounds(Hraw[r, q, s, a]) + @inbounds(Hraw[q, r, s, a])) / 2
    end)
    M = point_curvature(p, H, load_point_jacobian(Jpt, s, a), load_N(aN, a))
    for q in 1:10, r in 1:10
        @inbounds M_out[r, q, s, a] = M[r, q]
    end
end

"Column q of B_s = M_s W_s and of V_s = J_s W_s."
@kernel function ut_point_bv_kernel!(B_out, V_out, @Const(M_buf), @Const(Jpt), @Const(W), p::UTParams)
    index = @index(Global, Linear)
    q, point = (index - 1) % NTHETA + 1, (index - 1) ÷ NTHETA
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

const ASSEMBLY_TILE = 8                                   # 56 = 7 tiles of 8
const ASSEMBLY_TILES = NTHETA ÷ ASSEMBLY_TILE
const ASSEMBLY_TILE_PAIRS = ASSEMBLY_TILES * (ASSEMBLY_TILES + 1) ÷ 2   # lower-triangle tile pairs

"""
Hessian assembly with one work-group per (lower-triangle tile pair, arc): the 10 x 8
slices of W_s and B_s are staged in shared memory once per sigma point and reused by
the 64 threads of the tile, each summing over s, then r.
"""
@kernel function ut_hessian_assemble_tiled_kernel!(hess, @Const(W), @Const(B), @Const(Vbar), @Const(aN), @Const(zeta),
        p::UTParams)
    group = @index(Group, Linear)
    local_index = @index(Local, Linear)
    Wt = @localmem Float64 (10, ASSEMBLY_TILE)
    Bt = @localmem Float64 (10, ASSEMBLY_TILE)
    tile_pair, a = (group - 1) % ASSEMBLY_TILE_PAIRS + 1, (group - 1) ÷ ASSEMBLY_TILE_PAIRS + 1
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
                @inbounds Wt[r, column] = W[r, ASSEMBLY_TILE * (tp - 1) + column, s, a]
            else
                @inbounds Bt[r, column] = B[r, ASSEMBLY_TILE * (tq - 1) + column, s, a]
            end
        end
        @synchronize
        for r in 1:10
            total += @inbounds(Wt[r, lp]) * @inbounds(Bt[r, lq])
        end
        @synchronize
    end
    if pp >= qq
        @inbounds hess[lower_position(pp, qq), a] = hessian_entry(total, Vbar, aN, zeta, pp, qq, a)
    end
end

# ---------------------------------------------------------------- callbacks

function ut_values!(out, x, ut::UTOracle, p::UTParams, n)
    d, w = ut.data, ut.workspace
    launch!(ut_forward_kernel!, n * p.n_points, w.Y, x, d.theta_index, d.L0, d.arc_std, d.gain_scale, d.steps, p)
    launch!(ut_moments_kernel!, n * NUT_ROWS, out, w.ybar, w.Y, d.inv_next, n, p)
    return nothing
end

function launch_factor!(x, ut::UTOracle, p::UTParams, n)
    d, w, chunk = ut.data, ut.workspace, ut.factor_chunk
    launch!(ut_factor_kernel!, n * (NL ÷ chunk_width(chunk)), w.S, w.dS, x, d.theta_index, d.L0, d.arc_std, p, chunk)
end

function launch_point_inputs!(x, ut::UTOracle, p::UTParams, n)
    d, w = ut.data, ut.workspace
    launch!(ut_point_input_kernel!, n * p.n_points * (NTHETA ÷ chunk_width(GPU_ARC_CHUNK)), w.Z, w.W, x,
        d.theta_index, d.L0, w.S, w.dS, d.gain_scale, d.steps, p, GPU_ARC_CHUNK)
end

function ut_jacobian!(vals, x, ut::UTOracle, p::UTParams, n)
    d, w = ut.data, ut.workspace
    P = p.n_points
    launch_factor!(x, ut, p, n)
    if ut.gpu
        launch_point_inputs!(x, ut, p, n)
        launch!(ut_point_step_kernel!, n * P, w.Y, w.Jpt, w.Z, d.steps, p)
        launch!(ut_point_v_kernel!, n * P * NTHETA, w.V, w.Jpt, w.W, p)
    else
        launch!(ut_point_jacobian_kernel!, n * P, w.Y, w.V, x, d.theta_index, d.L0, w.S, w.dS, d.gain_scale, d.steps, p)
    end
    launch!(ut_reduce_kernel!, n * NX * (NTHETA + 1), w.ybar, w.Vbar, w.Y, w.V, p)
    launch!(ut_jacobian_kernel!, n * NUT_ROWS * NTHETA, w.jac, w.Y, w.ybar, w.V, w.Vbar, d.inv_next, p)
    gather!(vals, w.jac, d.jac_gather)
    return nothing
end

function ut_hessian!(vals, x, y, ut::UTOracle, p::UTParams, n)
    d, w = ut.data, ut.workspace
    P = p.n_points
    launch!(ut_forward_kernel!, n * P, w.Y, x, d.theta_index, d.L0, d.arc_std, d.gain_scale, d.steps, p)
    launch!(ut_hessian_prep_kernel!, n * (NX + NL), w.ybar, w.aN, w.Y, y, d.inv_next, n, p)
    launch_factor!(x, ut, p, n)
    if ut.gpu
        launch_point_inputs!(x, ut, p, n)
        launch!(ut_point_adjoint_kernel!, n * P, w.Hraw, w.Jpt, w.mu, w.Z, w.Y, w.ybar, w.aN, d.steps, p)
        launch!(ut_point_curvature_kernel!, n * P, w.M, w.Hraw, w.Jpt, w.aN, p)
        launch!(ut_point_bv_kernel!, n * P * NTHETA, w.B, w.V, w.M, w.Jpt, w.W, p)
    else
        launch!(ut_point_hessian_kernel!, n * P, w.W, w.B, w.V, w.mu, w.Y, w.ybar, w.aN, x, d.theta_index, d.L0,
            w.S, w.dS, d.gain_scale, d.steps, p)
    end
    launch!(ut_reduce_kernel!, n * NX * (NTHETA + 1), w.ybar, w.Vbar, w.Y, w.V, p)   # Vbar
    launch!(ut_zeta_kernel!, n * NHESS_ARC, w.zeta, w.mu, x, d.theta_index, d.L0, d.arc_std,
        d.gain_scale, d.steps, d.pair_p, d.pair_q, p)
    if ut.gpu
        launch!(ut_hessian_assemble_tiled_kernel!, n * ASSEMBLY_TILE_PAIRS * ASSEMBLY_TILE^2, w.hess, w.W, w.B,
            w.Vbar, w.aN, w.zeta, p; workgroup = ASSEMBLY_TILE^2)
    else
        launch!(ut_hessian_assemble_kernel!, n * NHESS_ARC, w.hess, w.W, w.B, w.Vbar, w.aN, w.zeta,
            d.pair_p, d.pair_q, p)
    end
    gather!(vals, w.hess, d.hess_gather)
    return nothing
end

# -------------------------------------------------------------------- setup

"""
    ut_oracle(n, offsets, scaling, seed, steps, p, backend) -> UTOracle

Index maps, sparsity and workspace of the UT oracle on `backend` (`nothing`: CPU);
`offsets` are the local-vector offsets of the means, feedforward, gain and factor blocks.
"""
function ut_oracle(n::Int, offsets, scaling::CovarianceScaling, seed::InitialGuess, steps, p::UTParams, backend)
    theta_index = zeros(Int, NTHETA, n)
    for a in 1:n
        theta_index[1:7, a] = offsets.means .+ 7(a-1) .+ (1:7)
        theta_index[8:10, a] = offsets.ff .+ 3(a-1) .+ (1:3)
        theta_index[11:28, a] = offsets.gains .+ 18(a-1) .+ (1:18)
        a >= 2 && (theta_index[29:56, a] = offsets.chol .+ 28(a-2) .+ (1:28))
    end
    active(a) = a == 1 ? 28 : NTHETA
    pair_p = [pp for pp in 1:NTHETA for _ in 1:pp]
    pair_q = [qq for pp in 1:NTHETA for qq in 1:pp]
    jac_rows, jac_cols, jac_gather = Int[], Int[], Int[]
    for a in 1:n, row in 1:NUT_ROWS, q in 1:active(a)
        push!(jac_rows, row <= NX ? row + 7(a-1) : 7n + (row - NX) + 28(a-1))
        push!(jac_cols, theta_index[q, a])
        push!(jac_gather, row + NUT_ROWS * (q - 1) + NUT_ROWS * NTHETA * (a - 1))
    end
    hess_rows, hess_cols, hess_gather = Int[], Int[], Int[]
    for a in 1:n, pp in 1:active(a), qq in 1:pp
        push!(hess_rows, theta_index[pp, a])
        push!(hess_cols, theta_index[qq, a])
        push!(hess_gather, lower_position(pp, qq) + NHESS_ARC * (a - 1))
    end
    device(v) = ExaModels.convert_array(v, backend)
    data = UTOracleData(device(theta_index), device(scaling.std[:, 1:n]), device(scaling.inv_std[:, 2:n+1]),
        device(copy(seed.gain_scale)), device(collect(steps)), device(copy(scaling.initial_factor)),
        device(pair_p), device(pair_q), device(jac_gather), device(hess_gather))
    gpu = backend !== nothing
    P, m = p.n_points, gpu ? n : 0
    buffer(dims...) = device(zeros(dims...))
    workspace = UTWorkspace(buffer(NL, n), buffer(NL, NL, n), buffer(NX, P, n), buffer(NX, n),
        buffer(10, NTHETA, P, n), buffer(NX, NTHETA, P, n), buffer(10, NTHETA, P, n), buffer(10, P, n),
        buffer(NX, NTHETA, n), buffer(NTHETA, n), buffer(NUT_ROWS, NTHETA, n), buffer(NHESS_ARC, n),
        buffer(NHESS_ARC, n), buffer(10, P, m), buffer(NX, 10, P, m), buffer(10, 10, P, m), buffer(10, 10, P, m))
    return UTOracle(data, workspace, (; jac_rows, jac_cols, hess_rows, hess_cols),
        gpu ? GPU_FACTOR_CHUNK : Val(NL), gpu)
end
