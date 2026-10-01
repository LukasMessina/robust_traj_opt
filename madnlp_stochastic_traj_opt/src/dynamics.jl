# Nondimensional rotating-frame CR3BP with thrust and variable mass, and the
# fixed-step explicit Runge-Kutta maps of the JAX script: Diffrax Dopri8 for the
# deterministic arcs and the XMDS2 RK9 tableau for the stochastic arcs. All
# functions are generic in the number type: the oracles evaluate them with
# Float64 values and with ForwardDiff dual numbers.

"Same arithmetic as `vector_field` in the JAX script."
@inline function cr3bp_vector_field(state::SVector{7}, control::SVector{3}, magnitude, mu, ve)
    x, y, z, vx, vy, vz, mass = state
    r1 = sqrt((x + mu)^2 + y * y + z * z)
    r2 = sqrt((x - 1 + mu)^2 + y * y + z * z)
    gravity = (1 - mu) / r1^3 + mu / r2^3
    return SVector(vx, vy, vz,
        2vy + x - (1 - mu) * (x + mu) / r1^3 - mu * (x - 1 + mu) / r2^3 + control[1] / mass,
        -2vx + y - gravity * y + control[2] / mass,
        -gravity * z + control[3] / mass,
        -magnitude / ve)
end

"Gradients of `w' f` with respect to the state, the control vector and the mass-flow magnitude."
@inline function cr3bp_vjp(state::SVector{7}, control::SVector{3}, magnitude, w::SVector{7}, mu, ve)
    x, y, z, vx, vy, vz, mass = state
    d1 = SVector(x + mu, y, z)
    d2 = SVector(x - 1 + mu, y, z)
    r1 = sqrt(d1[1]^2 + y * y + z * z)
    r2 = sqrt(d2[1]^2 + y * y + z * z)
    a = SVector(w[4], w[5], w[6])
    # Gravity gradient: d/dr of -(1-mu) d1/r1^3 - mu d2/r2^3, contracted with a.
    g1 = (1 - mu) / r1^3
    g2 = mu / r2^3
    s1 = 3 * (1 - mu) * dot(a, d1) / r1^5
    s2 = 3 * mu * dot(a, d2) / r2^5
    position = -(g1 + g2) * a + s1 * d1 + s2 * d2
    state_bar = SVector(position[1] + w[4], position[2] + w[5], position[3],
        w[1] - 2w[5], w[2] + 2w[4], w[3], -dot(a, control) / mass^2)
    return state_bar, a / mass, -w[7] / ve
end

abstract type RKTableau end
struct Dopri8 <: RKTableau end
struct XMDS2RK9 <: RKTableau end
tableau(::Type{Dopri8}) = (DOPRI8_A, DOPRI8_B)
tableau(::Type{XMDS2RK9}) = (XMDS2_RK9_A, XMDS2_RK9_B)

"Stages whose increments reach the solution (Dopri8's FSAL stage does not)."
function needed_stages(A, B)
    S = length(B)
    need = falses(S)
    for j in S:-1:1
        need[j] = B[j] != 0 || any(i -> need[i] && A[i][j] != 0, j+1:S)
    end
    return need
end

_stage(i) = Symbol(:Y, i)
_increment(i) = Symbol(:k, i)

function _rk_forward_body(TB)
    A, B = tableau(TB)
    need = needed_stages(A, B)
    body = Expr[]
    for i in eachindex(B)
        need[i] || continue
        terms = [:($(A[i][j]) * $(_increment(j))) for j in 1:i-1 if A[i][j] != 0]
        push!(body, :($(_stage(i)) = $(isempty(terms) ? :y0 : :(y0 + $(Expr(:call, :+, terms...))))))
        # Diffrax: k_i = f(Y_i) dt + G dW, with the control increment of the whole step.
        push!(body, :($(_increment(i)) = h * cr3bp_vector_field($(_stage(i)), control, magnitude, mu, ve) + gdw))
    end
    terms = [:($(B[j]) * $(_increment(j))) for j in eachindex(B) if B[j] != 0]
    push!(body, :(y1 = y0 + $(Expr(:call, :+, terms...))))
    return body, Tuple(_stage(i) for i in eachindex(B) if need[i])
end

"""
One explicit RK step `y1 = y0 + sum_j b_j k_j`, `k_j = h f(Y_j) + gdw`, with the
control and its magnitude held over the step (zero-order hold).
"""
@generated function rk_step(::Type{TB}, y0::SVector{7}, control::SVector{3}, magnitude,
        gdw::SVector{7}, h, mu, ve) where {TB<:RKTableau}
    body, _ = _rk_forward_body(TB)
    return quote
        Base.@_inline_meta
        $(body...)
        return y1
    end
end

"Like `rk_step`, but also returns the stage states used by `rk_adjoint`."
@generated function rk_step_stages(::Type{TB}, y0::SVector{7}, control::SVector{3}, magnitude,
        gdw::SVector{7}, h, mu, ve) where {TB<:RKTableau}
    body, stages = _rk_forward_body(TB)
    return quote
        Base.@_inline_meta
        $(body...)
        return y1, ($(stages...),)
    end
end

"""
Discrete adjoint of `rk_step`: given `g = dl/dy1`, returns `dl/dy0`, `dl/dcontrol`
and `dl/dmagnitude` (`gdw` constant). Evaluated with dual-number stages it is the
forward-over-reverse second derivative of `g' y1`.
"""
@generated function rk_adjoint(::Type{TB}, stages::Tuple, control::SVector{3}, magnitude,
        g::SVector{7}, h, mu, ve) where {TB<:RKTableau}
    A, B = tableau(TB)
    need = needed_stages(A, B)
    position = Dict(i => p for (p, i) in enumerate(findall(need)))
    stage_bar(i) = Symbol(:Ybar, i)
    body = Expr[:(control_bar = zero(control) * zero(eltype(stages[1]))),
                :(magnitude_bar = zero(magnitude) * zero(eltype(stages[1])))]
    for j in reverse(findall(need))
        terms = Any[]
        B[j] != 0 && push!(terms, :($(B[j]) * g))
        for i in j+1:length(B)
            need[i] && A[i][j] != 0 && push!(terms, :($(A[i][j]) * $(stage_bar(i))))
        end
        increment_bar = length(terms) == 1 ? terms[1] : Expr(:call, :+, terms...)
        push!(body, quote
            sb, cb, mb = cr3bp_vjp(stages[$(position[j])], control, magnitude, $increment_bar, mu, ve)
            $(stage_bar(j)) = h * sb
            control_bar = control_bar + h * cb
            magnitude_bar = magnitude_bar + h * mb
        end)
    end
    push!(body, :(state_bar = g + $(Expr(:call, :+, (stage_bar(j) for j in findall(need))...))))
    return quote
        Base.@_inline_meta
        $(body...)
        return state_bar, control_bar, magnitude_bar
    end
end

"Velocity-channel additive diffusion `G dW` with G = [0; sigma I3; 0]."
@inline diffusion_increment(sigma, dw::SVector{3,T}) where {T} =
    SVector{7,T}(zero(T), zero(T), zero(T), sigma * dw[1], sigma * dw[2], sigma * dw[3], zero(T))

"JAX `integration_map` with one substep: Dopri8 on t in [0, 1] with rhs `duration * f`."
deterministic_step(state, control, magnitude, duration, mu, ve) =
    rk_step(Dopri8, state, control, magnitude, zero(SVector{7,eltype(state)}), duration, mu, ve)
