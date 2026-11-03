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
