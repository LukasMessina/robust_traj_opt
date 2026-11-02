# Unscented transform and the stochastic arc map. The literal `propagation_arc`
# follows `build_propagation_arc_map(...).arc` of the JAX script step by step; it
# builds the seed, evaluates diagnostics and is the reference for the oracles.
#
# The state spread is the node factor itself, S = sqrt(d) diag(std) L, so that
# S S' = d P exactly. The JAX script refactorizes chol(d P + jitter I) instead;
# both agree to O(jitter) = 1e-12, but the refactorization divides its derivatives
# by the factor diagonal (and by sqrt(jitter) when a diagonal collapses), while the
# direct spread is linear in L.

"Unscented weights with kappa = 0 (the reference): the center point has weight zero."
function unscented_weights(dimension)
    w = fill(1.0 / (2.0 * dimension), 2dimension + 1)
    w[1] = 0.0
    return w
end

psi_inverse(dimension, beta) = sqrt(quantile(Chisq(dimension), 1.0 - beta))

"""
Normalization of state deviations: a normalized deviation e corresponds to the state
deviation `to_state(scale, e)`. `DiagonalScale` divides each component by a standard
deviation (single-phase cases, whose initial covariance is diagonal); `TriangularScale`
uses a lower Cholesky factor T of the initial covariance in state units, so that the
normalized initial covariance is the identity including its off-diagonal terms.
"""
struct DiagonalScale
    scale::SVector{7,Float64}
end

struct TriangularScale
    T::SMatrix{7,7,Float64,49}
    Tinv::SMatrix{7,7,Float64,49}
end

TriangularScale(T::AbstractMatrix) = TriangularScale(SMatrix{7,7}(LowerTriangular(T)), SMatrix{7,7}(inv(LowerTriangular(T))))

@inline to_state(s::DiagonalScale, e) = s.scale .* e
@inline to_state(s::TriangularScale, e) = s.T * e
@inline to_normalized(s::DiagonalScale, v) = v ./ s.scale
@inline to_normalized(s::TriangularScale, v) = s.Tinv * v
"Weight T^-T B T^-1 of normalized deviations for a symmetric weight B of state deviations."
@inline normalized_weight(s::DiagonalScale, B::SMatrix{7,7}) = B ./ (s.scale * s.scale')
@inline normalized_weight(s::TriangularScale, B::SMatrix{7,7}) = s.Tinv' * B * s.Tinv
"Normalized-to-state transform as a dense matrix."
state_transform(s::DiagonalScale) = Matrix(Diagonal(Vector(s.scale)))
state_transform(s::TriangularScale) = Matrix(s.T)

"""
Everything the arc map needs, as an isbits value that is also passed to the
device kernels. Column classes of the augmented spread (reference model):
1..7 state, 8..13 navigation, 14..16 Wiener increments. Each outer point carries
`inner` executed controls: the six Gates cubature points, or one (the command
itself) on arcs without thrust (`tmax == 0`), where the six would coincide.
The control type `C` (`CartesianControls`, `PolarControls` or `ThrottleControls`) fixes the
feedforward inputs of the oracles and, through `ntheta`, their number at compile time.
"""
struct UTParams{G<:GravityModel,S,C}
    d::Int
    inner::Int                    # executed-control points per outer point: 6, or 1 without thrust
    n_points::Int                 # 2d outer points times `inner`
    w_point::Float64              # outer weight 1/(2d) times inner weight 1/inner
    ct::Float64                   # 1/(d + kappa)
    scale::S                      # DiagonalScale or TriangularScale
    nav_factor::SMatrix{6,6,Float64,36}  # lower factor of d times the normalized navigation covariance
    nav_cov::SMatrix{6,6,Float64,36}     # normalized navigation covariance
    tmax::Float64
    gravity::G
    ve::Float64
    sigma_nd::Float64
    jitter::Float64               # added to the Gates covariance before its Cholesky factor
    var_mag::Float64              # sigma_2^2
    var_point::Float64            # sigma_4^2
    sqrt_d::Float64
    sqrt3::Float64
    psi::Float64
    eps::Float64                  # control_norm_eps
    floor::Float64                # spectral_radius_floor
    smoothing_sq::Float64         # eigenvalue_smoothing^2
    inv_sigma_sq::Float64         # state_covariance_weight / bryson_sigma_factor^2
    flow_sq::Float64              # (T_max mass_flow_smoothing)^2; 0: the reference's ||u||
    controls::C                   # feedforward inputs; see `model_params`
end

"Feedforward inputs S (the reference)."
struct CartesianControls end
"Polar feedforward inputs (Gamma, alpha, beta): S = Gamma H(n) (cos a cos b, sin a cos b, sin b)."
struct PolarControls end
"""
Throttle and direction as in the deterministic optimisation: d in the f inputs, Gamma the 57th input, S = Gamma d.
`ThrottleControls{true}` (`sigma_command_lifting`) also lifts the commands of the outer points that carry
feedback (`lift_index`): c_o = eta_o e_o with ||e_o|| = 1 (eta_o free) are the oracle inputs
LIFT_BASE + 4(l-1) + (1:4), and the model ties them to Gamma d + sigma K m_o by equality rows.
"""
struct ThrottleControls{Lifted} end

const CONTROL_TYPES = Dict(:cartesian => CartesianControls(), :polar => PolarControls(), :throttle => ThrottleControls{false}())

const NLIFT = 4NP        # lifted outer points: the + and - points of the 6 state and 6 navigation feedback columns
const LIFT_BASE = 57     # (eta_l, e_l) are the oracle inputs LIFT_BASE + 4(l-1) + (1:4)

"Number of arc inputs of the UT oracle (m 7, f 3, K_hat 18, L_tilde 28, the throttle and the lifted commands)."
@inline ntheta(::UTParams) = 56
@inline ntheta(::UTParams{<:GravityModel,<:Any,<:ThrottleControls}) = 57
@inline ntheta(::UTParams{<:GravityModel,<:Any,ThrottleControls{true}}) = LIFT_BASE + 4NLIFT
@inline lifted(::UTParams) = false
@inline lifted(::UTParams{<:GravityModel,<:Any,ThrottleControls{true}}) = true

"""
Lift index 1..NLIFT of outer point `o` when its command carries feedback (state columns 1..6 and navigation
columns 8..13; the mass column 7 has no position/velocity entries), 0 otherwise (c_o = S).
"""
@inline function lift_index(p::UTParams, o::Int)
    j = (o - 1) % p.d + 1
    column = j <= NP ? j : NX < j <= NX + NP ? j - 1 : 0
    return column == 0 ? 0 : column + 2NP * (o > p.d)
end
@inline nhess(p::UTParams) = ntheta(p) * (ntheta(p) + 1) ÷ 2
"Models without the norm smoothing: exact ||S||, plain mass flow, Gates factors without jitter."
@inline exact_norm(::UTParams) = true
@inline exact_norm(::UTParams{<:GravityModel,<:Any,CartesianControls}) = false

"""
UT parameters from the problem data: `scale` (a vector of standard deviations or a lower
factor of the initial covariance, in state units) converts normalized covariance coordinates
to state units; `nav_cov` is the normalized navigation covariance (a vector of variances or
a matrix); `tmax`, `ve` and `sigma_nd` are in the units of the propagated state.
"""
function ut_params(o::Options; scale, nav_cov, tmax, gravity, ve, sigma_nd)
    d = UT_DIMENSION
    inner = tmax == 0 ? 1 : 6
    # The reference constant is the literal 1e-30; keep it bitwise.
    smoothing_sq = o.eigenvalue_smoothing == JAX_EIGENVALUE_SMOOTHING ? 1e-30 : o.eigenvalue_smoothing^2
    normalization = scale isa AbstractVector ? DiagonalScale(SVector{7}(scale)) : TriangularScale(scale)
    covariance, factor = if nav_cov isa AbstractVector
        SMatrix{6,6}(Diagonal(SVector{6}(nav_cov))), SMatrix{6,6}(Diagonal(SVector{6}(sqrt.(max.(d .* nav_cov, 0.0)))))
    else
        C = SMatrix{6,6}((nav_cov + nav_cov') / 2)
        C, SMatrix{6,6}(cholesky(Symmetric(Matrix(d * C))).L)
    end
    return UTParams(d, inner, 2d * inner, unscented_weights(d)[2] * (1.0 / inner), 1.0 / d,
        normalization, factor, covariance,
        tmax, gravity, ve, sigma_nd, o.cholesky_jitter,
        o.gates_proportional_magnitude_std^2, o.gates_proportional_pointing_std^2,
        sqrt(Float64(d)), sqrt(3.0), psi_inverse(NU, o.violation_parameter),
        o.control_norm_eps, o.spectral_radius_floor, smoothing_sq, o.state_covariance_weight / o.bryson_sigma_factor^2,
        (tmax * o.mass_flow_smoothing)^2, CartesianControls())
end

"""
UT parameters of the stochastic model. Polar and throttle controls (S = Gamma d, ||d|| = 1) use the
plain mass-flow norm (delta = 0) and Gates factors without jitter: the Gates factor of a command
without feedback is Gamma chol(Q_G(d)) and its executed points Gamma (d +- sqrt(3) G_l), so their mass
flow is Gamma times a smooth function of d (||d +- sqrt(3) G_l|| >= 1 - sqrt(3) sigma_max). The nominal
and seed solves keep the parameters of `ut_params`. `model_mass_flow_smoothing` (a diagnostic, default 0)
puts delta back in the sigma-point mass flow only; the feedforward and the Gates factors stay exact.
`sigma_command_lifting` lifts the feedback commands of the thrust arcs (`ThrottleControls{true}`).
"""
function model_params(o::Options, p::UTParams)
    o.control_parameterization == :cartesian && return p
    controls = o.sigma_command_lifting && p.tmax != 0 ? ThrottleControls{true}() : CONTROL_TYPES[o.control_parameterization]
    flow_sq = (p.tmax * o.model_mass_flow_smoothing)^2
    return UTParams(ntuple(i -> fieldname(UTParams, i) === :flow_sq ? flow_sq : fieldname(UTParams, i) === :controls ?
        controls : getfield(p, i), fieldcount(UTParams))...)
end
model_params(o::Options, p::AbstractVector{<:UTParams}) = [model_params(o, q) for q in p]

ut_params(case::CaseData, o::Options, n::Normalization) = ut_params(o; scale = n.scale,
    nav_cov = navigation_variances(case, o, n), tmax = case.tmax, gravity = case.gravity, ve = case.ve,
    sigma_nd = acceleration_diffusion_nd(case, o))

@inline control_norm(f::SVector{3}, eps) = sqrt(dot(f, f) + eps^2)

"||S|| as the model evaluates it: smoothed by eps, or exact for the polar and throttle controls (||S|| = Gamma)."
@inline feedforward_norm(p, f::SVector{3}) = exact_norm(p) ? norm(f) : control_norm(f, p.eps)

"Reflection H(n) u that takes e1 to the unit seed direction n (the identity for n = e1)."
@inline function reflect_e1(u::SVector{3}, n::SVector{3})
    w = SVector(n[1] - 1.0, n[2], n[3])
    ww = dot(w, w)
    return ww < 1e-24 ? u : u - (2 * dot(w, u) / ww) * w
end

"Unit direction d = H(n) (cos a cos b, sin a cos b, sin b)."
@inline polar_direction(alpha, beta, n::SVector{3}) =
    reflect_e1(SVector(cos(alpha) * cos(beta), sin(alpha) * cos(beta), sin(beta)), n)

"""
Feedforward of the f inputs `theta`: S itself (cartesian), Gamma d(alpha, beta) (polar, seed direction n)
or Gamma d (throttle `gamma`, the direction d in `theta`).
"""
@inline feedforward_command(::UTParams, theta::SVector{3}, n, gamma) = theta
@inline feedforward_command(::UTParams{<:GravityModel,<:Any,PolarControls}, theta::SVector{3}, n, gamma) =
    theta[1] * polar_direction(theta[2], theta[3], n)
@inline feedforward_command(::UTParams{<:GravityModel,<:Any,<:ThrottleControls}, theta::SVector{3}, n, gamma) = gamma * theta

"""
Jacobian J = dS/d(Gamma, alpha, beta) of the polar feedforward and its second derivatives
(packed lower: GG, aG, aa, bG, ba, bb, each a 3-vector).
"""
@inline function polar_derivatives(theta::SVector{3}, n::SVector{3})
    G, a, b = theta[1], theta[2], theta[3]
    ca, sa, cb, sb = cos(a), sin(a), cos(b), sin(b)
    d = reflect_e1(SVector(ca * cb, sa * cb, sb), n)
    da = reflect_e1(SVector(-sa * cb, ca * cb, 0.0), n)
    db = reflect_e1(SVector(-ca * sb, -sa * sb, cb), n)
    daa = reflect_e1(SVector(-ca * cb, -sa * cb, 0.0), n)
    dab = reflect_e1(SVector(sa * sb, -ca * sb, 0.0), n)
    dbb = reflect_e1(SVector(-ca * cb, -sa * cb, -sb), n)
    J = hcat(d, G * da, G * db)
    return J, (zero(d), da, G * daa, db, G * dab, G * dbb)
end

"""
Mass-flow magnitude of an executed sigma point, sqrt(||u||^2 + (T_max delta)^2), the smoothing
that `control_norm` applies to the feedforward (delta = `mass_flow_smoothing`). The reference
arc map uses the plain ||u|| (delta = 0), whose curvature grows like 1/||u|| on coast arcs.
Arcs without thrust have exactly zero mass flow.
"""
@inline sigma_mass_flow(p, u) = sigma_mass_flow(p, u, p.flow_sq)
@inline sigma_mass_flow(p, u, flow_sq) = p.tmax == 0 ? zero(eltype(u)) : sqrt(dot(u, u) + flow_sq)

"""
delta^2 term of the mass flow of outer point `o` (0: the zero-weight center). The polar and throttle models
(`model_mass_flow_smoothing`) smooth only the points whose command carries feedback (`lift_index`): the others
command Gamma d, whose mass flow Gamma ||d +- sqrt(3) G(d)_l|| is already smooth, and keep the exact norm. The
cartesian reference smooths every point.
"""
@inline outer_flow_sq(p::UTParams, o::Int) = exact_norm(p) && (o == 0 || lift_index(p, o) == 0) ? 0.0 : p.flow_sq

"Q_G(u) = sigma_4^2 ||u||^2 I + (sigma_2^2 - sigma_4^2) u u'."
@inline gates_covariance(c::SVector{3}, var_mag, var_point) =
    var_point * dot(c, c) * SMatrix{3,3}(1.0I) + (var_mag - var_point) * (c * c')

"""
Cholesky factor of the six-point Gates cubature conditional on the command `c`. The polar and
throttle models leave out the jitter, so the factor is homogeneous of degree one in `c` (Gamma G(d)
without feedback).
"""
@inline function gates_factor(c::SVector{3}, p::UTParams)
    covariance = gates_covariance(c, p.var_mag, p.var_point) + (exact_norm(p) ? 0.0 : p.jitter) * SMatrix{3,3}(1.0I)
    return lower_matrix3(cholesky_packed(covariance, -Inf))
end

"Smoothed analytic 3x3 largest eigenvalue, `spectral_radius` of the JAX script."
@inline spectral_radius(T::SMatrix{3,3}, floor, smoothing_sq) =
    sqrt(largest_eigenvalue(T, smoothing_sq) + floor^2)

"Trigonometric 3x3 symmetric largest eigenvalue, smoothed by `s` in `p = sqrt(p2/6 + s^2)`."
@inline function largest_eigenvalue(T::SMatrix{3,3}, smoothing_sq)
    q = (T[1, 1] + T[2, 2] + T[3, 3]) / 3
    p1 = T[1, 2]^2 + T[1, 3]^2 + T[2, 3]^2
    p2 = (T[1, 1] - q)^2 + (T[2, 2] - q)^2 + (T[3, 3] - q)^2 + 2p1
    p = sqrt(p2 / 6 + smoothing_sq)
    b = (T - q * SMatrix{3,3}(1.0I)) / p
    det = b[1, 1] * (b[2, 2] * b[3, 3] - b[2, 3] * b[3, 2]) -
          b[1, 2] * (b[2, 1] * b[3, 3] - b[2, 3] * b[3, 1]) +
          b[1, 3] * (b[2, 1] * b[3, 2] - b[2, 2] * b[3, 1])
    r = clamp(det / 2, -1 + 1e-12, 1 - 1e-12)
    return q + 2p * cos(acos(r) / 3)
end

"""
Executed-control covariance of the nested UT in closed form. With the spread S
(S S' = d P) and the Gates factor G (G G' = Q_G(c) + eps I), the weighted sum over all
192 executed points of the arc map is exactly

    T = ct [(1 + s2 - s4) K M K' + s4 tr(K M K') I] + Q_G(f) + eps I,
    M = (S S')[1:6,1:6] + d R_nav = d (P6 + R_nav).
"""
@inline function control_covariance(p::UTParams, f::SVector{3}, K::SMatrix{3,6}, P6::SMatrix{6,6})
    A = commanded_control_covariance(p, K, P6)
    trA = A[1, 1] + A[2, 2] + A[3, 3]
    return (1 + p.var_mag - p.var_point) * A + p.var_point * trA * SMatrix{3,3}(1.0I) +
           gates_covariance(f, p.var_mag, p.var_point) + p.jitter * SMatrix{3,3}(1.0I)
end

"Covariance of feedback commands: navigation included, execution error excluded."
@inline function commanded_control_covariance(p::UTParams, K::SMatrix{3,6}, P6::SMatrix{6,6})
    M = p.d * P6 + p.d * p.nav_cov
    return p.ct * (K * M * K')
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

"The same with a full lower node scale S: (S6 L6)(S6 L6)', S6 and L6 the 6x6 blocks."
@inline function position_velocity_covariance(S::SMatrix{7,7}, L21::SVector{21,T}) where {T}
    F = SMatrix{6,6,T}(ntuple(Val(36)) do linear
        r, c = (linear - 1) % 6 + 1, (linear - 1) ÷ 6 + 1
        total = zero(T)
        for m in c:r
            total += S[r, m] * L21[packed_index(m, c)]
        end
        r >= c ? total : zero(T)
    end)
    return F * F'
end

"Chance-constraint value ||S_tilde||_eps + psi rho(T) (must be <= 1)."
@inline control_chance(p::UTParams, f::SVector{3}, K::SMatrix{3,6}, P6::SMatrix{6,6}) =
    control_norm(f, p.eps) + control_spread(p, f, K, P6)

"Covariance part psi rho(T) of the thrust chance left-hand side."
@inline control_spread(p::UTParams, f::SVector{3}, K::SMatrix{3,6}, P6::SMatrix{6,6}) =
    p.psi * spectral_radius(control_covariance(p, f, K, P6), p.floor, p.smoothing_sq)

"Packed UT state spread sqrt(d) diag(std) L of a node with scales `std` and factor `L`."
@inline spread_factor(p::UTParams, std::SVector{7}, L::SVector{28}) =
    SVector{28}(ntuple(r -> (p.sqrt_d * std[LTRI_ROWS[r]]) * L[r], Val(28)))

"Packed spread sqrt(d) S L of a full lower node scale S: entry (r, c) = sqrt(d) sum_{m=c..r} S_rm L_mc."
@inline function spread_factor(p::UTParams, S::SMatrix{7,7}, L::SVector{28})
    return SVector{28}(ntuple(Val(28)) do index
        r, c = LTRI_ROWS[index], LTRI_COLS[index]
        total = zero(eltype(L))
        for m in c:r
            total += S[r, m] * L[packed_index(m, c)]
        end
        p.sqrt_d * total
    end)
end

"""
    propagation_arc(p, mean, P, feedforward, gain, duration)

`(mean_next, P_next, T)`: UT prediction of the next node mean, its normalized
covariance and the normalized executed-control covariance, from the 33 outer and
nested executed points (one RK9 step per arc). The spread is sqrt(d) chol(P);
`propagation_arc_factor` takes a lower factor F of P (P = F F') instead.
"""
function propagation_arc(p::UTParams, mean::AbstractVector, P::AbstractMatrix, feedforward::AbstractVector,
        gain::AbstractMatrix, duration)
    F = lower_matrix(cholesky_packed(SMatrix{7,7}(P), p.jitter))
    return propagation_arc_factor(p, mean, F, feedforward, gain, duration)
end

function propagation_arc_factor(p::UTParams, mean::AbstractVector, F::AbstractMatrix, feedforward::AbstractVector,
        gain::AbstractMatrix, duration)
    T = promote_type(eltype(mean), eltype(F), eltype(feedforward), eltype(gain), typeof(duration))
    d = p.d
    spread = p.sqrt_d * SMatrix{7,7}(F)
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
            state_offset = to_state(p.scale, spread[:, j])
            control_offset = K * spread[SOneTo(6), j]
        elseif j <= NX + NP
            state_offset = zero(SVector{7,T})
            control_offset = K * p.nav_factor[:, j - NX]
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
        gdw = diffusion_increment(p.sigma_nd, sqrt(duration) * SVector{3}(xi[i]))
        points = if p.inner == 1
            (commanded[i],)
        else
            G = gates_factor(commanded[i], p)
            Tuple(commanded[i] + sign * (p.sqrt3 * G[:, l]) for sign in (1, -1) for l in 1:3)
        end
        for e in points
            u = p.tmax * e
            push!(weights, outer[i] * (1.0 / p.inner))
            push!(executed, p.tmax == 0 ? zero(e) : e)   # nothing is executed without thrust
            # delta = 0 keeps the reference's norm bitwise.
            flow_sq = outer_flow_sq(p, i - 1)
            magnitude = flow_sq == 0 && p.tmax != 0 ? norm(u) : sigma_mass_flow(p, u, flow_sq)
            push!(propagated, rk_step(XMDS2RK9, states[i], u, magnitude, gdw, duration, p.gravity, p.ve))
        end
    end
    mean_next = sum(weights .* propagated)
    deviations = [to_normalized(p.scale, y - mean_next) for y in propagated]
    covariance = sum(w * (dv * dv') for (w, dv) in zip(weights, deviations))
    mean_executed = sum(weights .* executed)
    control_cov = sum(w * ((e - mean_executed) * (e - mean_executed)') for (w, e) in zip(weights, executed))
    return mean_next, (covariance + covariance') / 2, (control_cov + control_cov') / 2
end

@inline arc_parameters(p::UTParams, k) = p
@inline arc_parameters(p::AbstractVector{<:UTParams}, k) = p[k]

"""
Forward propagation of the node covariance at fixed node means (JAX `propagate_stochastic_moments`).
`p` is one `UTParams` or one per arc; covariances are in the same normalized coordinates
at every node, so a change of origin between arcs leaves them unchanged.
"""
function propagate_stochastic_moments(p, means, feedforward, gains, P0, steps)
    covariances = [Matrix{Float64}(P0)]
    controls = Matrix{Float64}[]
    propagated = [Vector{Float64}(means[:, 1])]
    for k in eachindex(steps)
        mu, P, T = propagation_arc(arc_parameters(p, k), means[:, k], covariances[end], feedforward[:, k],
            gains[:, :, k], steps[k])
        push!(propagated, Vector(mu))
        push!(covariances, Matrix(P))
        push!(controls, Matrix(T))
    end
    return reduce(hcat, propagated), stack(covariances; dims=3), stack(controls; dims=3)
end
