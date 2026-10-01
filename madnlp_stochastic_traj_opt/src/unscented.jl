# Unscented transform and the stochastic arc map. The literal `propagation_arc`
# mirrors `build_propagation_arc_map(...).arc` of the JAX script step by step; it
# builds the seed, evaluates diagnostics and is the reference for the oracles.

"Unscented weights with kappa = 0 (the reference): the center point has weight zero."
function unscented_weights(dimension)
    w = fill(1.0 / (2.0 * dimension), 2dimension + 1)
    w[1] = 0.0
    return w
end

psi_inverse(dimension, beta) = sqrt(quantile(Chisq(dimension), 1.0 - beta))

"""
Everything the arc map needs, as an isbits value that is also passed to the
device kernels. Column classes of the augmented spread (reference model):
1..7 state, 8..13 navigation, 14..16 Wiener increments.
"""
struct UTParams
    d::Int
    n_points::Int                 # 2d outer points times 6 Gates points
    w_point::Float64              # outer weight 1/(2d) times inner weight 1/6
    ct::Float64                   # 1/(d + kappa)
    scale::SVector{7,Float64}
    nav_root::SVector{6,Float64}  # sqrt(d * normalized navigation variance)
    nav_var::SVector{6,Float64}
    tmax::Float64
    mu::Float64
    ve::Float64
    sigma_nd::Float64
    jitter::Float64
    var_mag::Float64              # sigma_2^2
    var_point::Float64            # sigma_4^2
    sqrt_d::Float64
    sqrt3::Float64
    psi::Float64
    eps::Float64                  # control_norm_eps
    floor::Float64                # spectral_radius_floor
    smoothing_sq::Float64         # eigenvalue_smoothing^2
    inv_sigma_sq::Float64         # 1 / bryson_sigma_factor^2
    flow_sq::Float64              # (T_max mass_flow_smoothing)^2; 0: the reference's ||u||
end

function ut_params(case::CaseData, o::Options, n::Normalization)
    d = UT_DIMENSION
    nav_var = navigation_variances(case, o, n)
    # The reference constant is the literal 1e-30; keep it bitwise.
    smoothing_sq = o.eigenvalue_smoothing == JAX_EIGENVALUE_SMOOTHING ? 1e-30 : o.eigenvalue_smoothing^2
    return UTParams(d, 2d * 6, unscented_weights(d)[2] * (1.0 / 6.0), 1.0 / d,
        n.scale, SVector{6}(sqrt.(max.(d .* nav_var, 0.0))), SVector{6}(nav_var),
        case.tmax, case.mu, case.ve, acceleration_diffusion_nd(case, o), o.cholesky_jitter,
        o.gates_proportional_magnitude_std^2, o.gates_proportional_pointing_std^2,
        sqrt(Float64(d)), sqrt(3.0), psi_inverse(NU, o.violation_parameter),
        o.control_norm_eps, o.spectral_radius_floor, smoothing_sq, 1.0 / o.bryson_sigma_factor^2,
        (case.tmax * o.mass_flow_smoothing)^2)
end

@inline control_norm(f::SVector{3}, eps) = sqrt(dot(f, f) + eps^2)

"""
Mass-flow magnitude of an executed sigma point, sqrt(||u||^2 + (T_max delta)^2), the smoothing
that `control_norm` applies to the feedforward (delta = `mass_flow_smoothing`). The reference
arc map uses the plain ||u|| (delta = 0), whose curvature grows like 1/||u|| on coast arcs.
"""
@inline sigma_mass_flow(p, u) = sqrt(dot(u, u) + p.flow_sq)

"Q_G(u) = sigma_4^2 ||u||^2 I + (sigma_2^2 - sigma_4^2) u u'."
@inline gates_covariance(c::SVector{3}, var_mag, var_point) =
    var_point * dot(c, c) * SMatrix{3,3}(1.0I) + (var_mag - var_point) * (c * c')

"Cholesky factor of the six-point Gates cubature conditional on the command `c`."
@inline function gates_factor(c::SVector{3}, p::UTParams)
    covariance = gates_covariance(c, p.var_mag, p.var_point) + p.jitter * SMatrix{3,3}(1.0I)
    return lower_matrix3(cholesky_packed(covariance, -Inf))
end

"Smoothed analytic 3x3 largest eigenvalue, `spectral_radius` of the JAX script."
@inline function spectral_radius(T::SMatrix{3,3}, floor, smoothing_sq)
    q = (T[1, 1] + T[2, 2] + T[3, 3]) / 3
    p1 = T[1, 2]^2 + T[1, 3]^2 + T[2, 3]^2
    p2 = (T[1, 1] - q)^2 + (T[2, 2] - q)^2 + (T[3, 3] - q)^2 + 2p1
    p = sqrt(p2 / 6 + smoothing_sq)
    b = (T - q * SMatrix{3,3}(1.0I)) / p
    det = b[1, 1] * (b[2, 2] * b[3, 3] - b[2, 3] * b[3, 2]) -
          b[1, 2] * (b[2, 1] * b[3, 3] - b[2, 3] * b[3, 1]) +
          b[1, 3] * (b[2, 1] * b[3, 2] - b[2, 2] * b[3, 1])
    r = clamp(det / 2, -1 + 1e-12, 1 - 1e-12)
    return sqrt(q + 2p * cos(acos(r) / 3) + floor^2)
end

"""
Executed-control covariance of the nested UT in closed form. With the Cholesky
spread S (S S' = d P + eps I) and the Gates factor G (G G' = Q_G(c) + eps I), the
weighted sum over all 192 executed points of the JAX map is exactly

    T = ct [(1 + s2 - s4) K M K' + s4 tr(K M K') I] + Q_G(f) + eps I,
    M = (S S')[1:6,1:6] + d R_nav = d P6 + eps I + d R_nav.
"""
@inline function control_covariance(p::UTParams, f::SVector{3}, K::SMatrix{3,6}, P6::SMatrix{6,6})
    M = p.d * P6 + SMatrix{6,6}(Diagonal(SVector{6}(ntuple(i -> p.jitter + p.d * p.nav_var[i], Val(6)))))
    A = p.ct * (K * M * K')
    trA = A[1, 1] + A[2, 2] + A[3, 3]
    return (1 + p.var_mag - p.var_point) * A + p.var_point * trA * SMatrix{3,3}(1.0I) +
           gates_covariance(f, p.var_mag, p.var_point) + p.jitter * SMatrix{3,3}(1.0I)
end

"Normalized 6x6 covariance from node scales and the first 21 packed factor entries."
@inline function position_velocity_covariance(std::SVector{7}, L21::SVector{21,T}) where {T}
    z = zero(T)
    F = SMatrix{6,6,T}(ntuple(Val(36)) do linear
        r, c = (linear - 1) % 6 + 1, (linear - 1) ÷ 6 + 1
        r >= c ? std[r] * L21[packed_index(r, c)] : z
    end)
    return F * F'
end

"Chance-constraint value ||S_tilde||_eps + psi rho(T) (must be <= 1)."
@inline control_chance(p::UTParams, f::SVector{3}, K::SMatrix{3,6}, P6::SMatrix{6,6}) =
    control_norm(f, p.eps) + p.psi * spectral_radius(control_covariance(p, f, K, P6), p.floor, p.smoothing_sq)

"Spread `cholesky_lower(d P + jitter I, jitter)` as a packed static factor."
@inline arc_spread(p::UTParams, P::SMatrix{7,7}) =
    cholesky_packed(p.d * P + p.jitter * SMatrix{7,7}(1.0I), p.jitter)

"""
    propagation_arc(p, mean, P, feedforward, gain, duration)

`(mean_next, P_next, T)`: UT prediction of the next node mean, its normalized
covariance and the normalized executed-control covariance, from the 33 outer and
198 nested sigma points exactly as the JAX map computes them (one RK9 step per arc).
"""
function propagation_arc(p::UTParams, mean::AbstractVector, P::AbstractMatrix, feedforward::AbstractVector,
        gain::AbstractMatrix, duration)
    T = promote_type(eltype(mean), eltype(P), eltype(feedforward), eltype(gain), typeof(duration))
    d = p.d
    spread = lower_matrix(arc_spread(p, SMatrix{7,7}(P)))
    K = SMatrix{3,6}(gain)
    m = SVector{7}(mean)
    f = SVector{3}(feedforward)
    outer = unscented_weights(d)
    states = Vector{SVector{7,T}}(undef, 2d + 1)
    commanded = Vector{SVector{3,T}}(undef, 2d + 1)
    xi = [zeros(T, NW) for _ in 1:2d+1]
    states[1], commanded[1] = m, f
    for j in 1:d
        if j <= NX
            state_offset = p.scale .* spread[:, j]
            control_offset = K * spread[SOneTo(6), j]
        elseif j <= NX + NP
            state_offset = zero(SVector{7,T})
            control_offset = K[:, j - NX] * p.nav_root[j - NX]
        else
            state_offset = zero(SVector{7,T})
            control_offset = zero(SVector{3,T})
            xi[1 + j][j - NX - NP] = p.sqrt_d
            xi[1 + d + j][j - NX - NP] = -p.sqrt_d
        end
        states[1 + j], states[1 + d + j] = m + state_offset, m - state_offset
        commanded[1 + j], commanded[1 + d + j] = f + control_offset, f - control_offset
    end
    weights = T[]
    executed = SVector{3,T}[]
    propagated = SVector{7,T}[]
    for i in 1:2d+1
        G = gates_factor(commanded[i], p)
        gdw = diffusion_increment(p.sigma_nd, sqrt(duration) * SVector{3}(xi[i]))
        for sign in (1, -1), l in 1:3
            e = commanded[i] + sign * (p.sqrt3 * G[:, l])
            u = p.tmax * e
            push!(weights, outer[i] * (1.0 / 6.0))
            push!(executed, e)
            # delta = 0 keeps the reference's norm bitwise.
            magnitude = p.flow_sq == 0 ? norm(u) : sigma_mass_flow(p, u)
            push!(propagated, rk_step(XMDS2RK9, states[i], u, magnitude, gdw, duration, p.mu, p.ve))
        end
    end
    mean_next = sum(weights .* propagated)
    deviations = [(y - mean_next) ./ p.scale for y in propagated]
    covariance = sum(w * (dv * dv') for (w, dv) in zip(weights, deviations))
    mean_executed = sum(weights .* executed)
    control_cov = sum(w * ((e - mean_executed) * (e - mean_executed)') for (w, e) in zip(weights, executed))
    return mean_next, (covariance + covariance') / 2, (control_cov + control_cov') / 2
end

"Forward propagation of the node covariance at fixed node means (JAX `propagate_stochastic_moments`)."
function propagate_stochastic_moments(p::UTParams, means, feedforward, gains, P0, steps)
    covariances = [Matrix{Float64}(P0)]
    controls = Matrix{Float64}[]
    propagated = [Vector{Float64}(means[:, 1])]
    for k in eachindex(steps)
        mu, P, T = propagation_arc(p, means[:, k], covariances[end], feedforward[:, k], gains[:, :, k], steps[k])
        push!(propagated, Vector(mu))
        push!(covariances, Matrix(P))
        push!(controls, Matrix(T))
    end
    return reduce(hcat, propagated), stack(covariances; dims=3), stack(controls; dims=3)
end
