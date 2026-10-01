# Thrust chance constraint  ||S_tilde_k||_eps + psi rho(T_k) <= 1  as an ExaModels
# VectorNonlinearOracle (one row per arc). T_k is the closed form of the nested-UT
# executed-control covariance (`control_covariance`), a polynomial of
# (f, K_hat, L_tilde rows 1..6); rho is the smoothed trigonometric largest-eigenvalue
# formula of the reference. Exact derivatives come from single and pair-seeded duals.

const NCHANCE = 42                              # f 3, K_hat 18, L_tilde 21
const NCHANCE_PAIRS = NCHANCE * (NCHANCE + 1) ÷ 2

struct ChanceData{VI,MI,VF,MF}
    index::MI        # 42 x N global variable index (0: fixed data)
    arc_std::MF
    gain_scale::VF
    L0::VF
    pair_p::VI
    pair_q::VI
    jac_gather::VI
    hess_gather::VI
end

@inline function chance_inputs(x, index, L0, a)
    f = SVector{3}(ntuple(i -> @inbounds(x[index[i, a]]), Val(3)))
    k = SVector{18}(ntuple(i -> @inbounds(x[index[3+i, a]]), Val(18)))
    L = a == 1 ? SVector{21}(ntuple(i -> @inbounds(L0[i]), Val(21))) :
                 SVector{21}(ntuple(i -> @inbounds(x[index[21+i, a]]), Val(21)))
    return f, k, L
end

@inline chance_value(p::UTParams, std, gs, f, k, L) =
    control_chance(p, f, gs * SMatrix{3,6}(k), position_velocity_covariance(std, L))

@kernel function chance_value_kernel!(out, @Const(x), @Const(index), @Const(L0), @Const(arc_std), @Const(gain_scale), p::UTParams)
    a = @index(Global, Linear)
    f, k, L = chance_inputs(x, index, L0, a)
    @inbounds out[a] = chance_value(p, arc_column(arc_std, a), gain_scale[a], f, k, L)
end

@kernel function chance_jacobian_kernel!(jac, @Const(x), @Const(index), @Const(L0), @Const(arc_std), @Const(gain_scale), p::UTParams)
    i = @index(Global, Linear)
    q, a = (i - 1) % NCHANCE + 1, (i - 1) ÷ NCHANCE + 1
    f, k, L = chance_inputs(x, index, L0, a)
    fd = SVector{3}(ntuple(j -> single_seed(f[j], q == j), Val(3)))
    kd = SVector{18}(ntuple(j -> single_seed(k[j], q == 3 + j), Val(18)))
    Ld = SVector{21}(ntuple(j -> single_seed(L[j], q == 21 + j), Val(21)))
    value = chance_value(p, arc_column(arc_std, a), @inbounds(gain_scale[a]), fd, kd, Ld)
    @inbounds jac[q, a] = ForwardDiff.partials(value)[1]
end

@kernel function chance_hessian_kernel!(hess, @Const(x), @Const(y), @Const(index), @Const(L0), @Const(arc_std),
        @Const(gain_scale), @Const(pair_p), @Const(pair_q), p::UTParams)
    i = @index(Global, Linear)
    pair, a = (i - 1) % NCHANCE_PAIRS + 1, (i - 1) ÷ NCHANCE_PAIRS + 1
    pp, qq = @inbounds(pair_p[pair]), @inbounds(pair_q[pair])
    f, k, L = chance_inputs(x, index, L0, a)
    fd = SVector{3}(ntuple(j -> pair_seed(f[j], pp == j, qq == j), Val(3)))
    kd = SVector{18}(ntuple(j -> pair_seed(k[j], pp == 3 + j, qq == 3 + j), Val(18)))
    Ld = SVector{21}(ntuple(j -> pair_seed(L[j], pp == 21 + j, qq == 21 + j), Val(21)))
    value = chance_value(p, arc_column(arc_std, a), @inbounds(gain_scale[a]), fd, kd, Ld)
    @inbounds hess[pair, a] = y[a] * pair_value(value)
end

"Global variable indices of the 42 inputs of T_k (f, K_hat, L_tilde rows 1..6) per arc."
function chance_input_index(n::Int, offsets)
    index = zeros(Int, NCHANCE, n)
    for a in 1:n
        index[1:3, a] = offsets.ff .+ 3(a-1) .+ (1:3)
        index[4:21, a] = offsets.gains .+ 18(a-1) .+ (1:18)
        a >= 2 && (index[22:42, a] = offsets.chol .+ 28(a-2) .+ (1:21))
    end
    return index
end

"Number of variable inputs of arc `a` (the first arc's covariance is fixed data)."
chance_active_inputs(a) = a == 1 ? 21 : NCHANCE

function chance_oracle(n::Int, offsets, nvar::Int, scaling::CovarianceScaling, seed::InitialGuess,
        p::UTParams, backend)
    index = chance_input_index(n, offsets)
    active = chance_active_inputs
    pair_p = [pp for pp in 1:NCHANCE for _ in 1:pp]
    pair_q = [qq for pp in 1:NCHANCE for qq in 1:pp]
    jac_rows, jac_cols, jac_gather = Int[], Int[], Int[]
    hess_rows, hess_cols, hess_gather = Int[], Int[], Int[]
    for a in 1:n
        for q in 1:active(a)
            push!(jac_rows, a); push!(jac_cols, index[q, a]); push!(jac_gather, q + NCHANCE * (a - 1))
        end
        for pp in 1:active(a), qq in 1:pp
            push!(hess_rows, index[pp, a]); push!(hess_cols, index[qq, a])
            push!(hess_gather, lower_position(pp, qq) + NCHANCE_PAIRS * (a - 1))
        end
    end
    device(v) = ExaModels.convert_array(v, backend)
    d = ChanceData(device(index), device(scaling.std[:, 1:n]), device(copy(seed.gain_scale)),
        device(copy(scaling.initial_factor[1:21])), device(pair_p), device(pair_q), device(jac_gather),
        device(hess_gather))
    jac_buffer = device(zeros(NCHANCE, n))
    hess_buffer = device(zeros(NCHANCE_PAIRS, n))
    return ExaModels.VectorNonlinearOracle(; nvar, ncon=n,
        jac_rows, jac_cols, hess_rows, hess_cols,
        lcon=device(fill(-Inf, n)), ucon=device(ones(n)),
        f! = (out, x) -> launch!(chance_value_kernel!, n, out, x, d.index, d.L0, d.arc_std, d.gain_scale, p),
        jac! = function (vals, x)
            launch!(chance_jacobian_kernel!, n * NCHANCE, jac_buffer, x, d.index, d.L0, d.arc_std, d.gain_scale, p)
            gather!(vals, jac_buffer, d.jac_gather)
        end,
        hess! = function (vals, x, y)
            launch!(chance_hessian_kernel!, n * NCHANCE_PAIRS, hess_buffer, x, y, d.index, d.L0, d.arc_std,
                d.gain_scale, d.pair_p, d.pair_q, p)
            gather!(vals, hess_buffer, d.hess_gather)
        end)
end
