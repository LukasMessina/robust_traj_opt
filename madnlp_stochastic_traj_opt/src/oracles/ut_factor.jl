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
