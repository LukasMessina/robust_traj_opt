pack_lower(L) = [L[r, c] for (r, c) in LOWER]
function unpack_lower(entries)
    L = zeros(eltype(entries), NX, NX)
    for (j, (r, c)) in enumerate(LOWER)
        L[r, c] = entries[j]
    end
    return L
end

function system_matrices(o::Options)
    Ac = [0.0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0]
    Bc = [0.0 0; 0 0; 1 0; 0 1]
    A = Matrix{Float64}(I, NX, NX) + o.dt * Ac
    B = o.dt * Bc + o.dt^2 / 2 * Ac * Bc
    G = 0.01 * Matrix{Float64}(I, NX, NX)
    return (; Ac, Bc, A, B, G)
end

"Exact finite-horizon covariance integral for this double integrator (Ac^2 = 0)."
function accumulated_process_covariance(Ac, Qc, h)
    return h * Qc + h^2 / 2 * (Ac * Qc + Qc * Ac') + h^3 / 3 * Ac * Qc * Ac'
end

function calibrate_continuous_diffusion(Ac, Qd, h)
    # Invert the same finite-horizon Lyapunov operator as Python. Its integral
    # is evaluated exactly here because Ac is nilpotent of degree two.
    operator = zeros(NX^2, NX^2)
    for j in 1:NX^2
        E = zeros(NX, NX)
        E[j] = 1
        operator[:, j] = vec(accumulated_process_covariance(Ac, E, h))
    end
    Qc = reshape(operator \ vec(Qd), NX, NX)
    Qc = Matrix(Symmetric((Qc + Qc') / 2))
    Lc = Matrix(cholesky(Symmetric(Qc)).L)
    @assert maximum(abs, accumulated_process_covariance(Ac, Qc, h) - Qd) < 1e-15
    return Qc, Lc
end

"""
ShARK, specialized only to constant additive diffusion and double-integrator drift.

Diffrax tableau: a21=5/6, b=(2/5,3/5), aW=(0,5/6), aH=(1,1), bW=1, bH=0.
Stage states are y1=x+Lc*H, y2=x+(5h/6)f(y1,u)+(5/6)Lc*W+Lc*H.
References: https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_solver/shark.py
and Foster, dos Reis & Strange, https://arxiv.org/abs/2210.17543 .
The scalar functions are also traced by ExaModels; no RNG is called here.
"""
@inline function shark_position(p, v, u, gWp, gWv, gHv, h)
    f1 = v + gHv
    f2 = v + (5h / 6) * u + (5 / 6) * gWv + gHv
    return p + h * ((2 / 5) * f1 + (3 / 5) * f2) + gWp
end
@inline shark_velocity(v, u, gWv, h) = v + h * ((2 / 5) * u + (3 / 5) * u) + gWv

function shark_step(x, u, W, H, Lc, h)
    gW, gH = Lc * W, Lc * H
    return [shark_position(x[1], x[3], u[1], gW[1], gW[3], gH[3], h),
            shark_position(x[2], x[4], u[2], gW[2], gW[4], gH[4], h),
            shark_velocity(x[3], u[1], gW[3], h),
            shark_velocity(x[4], u[2], gW[4], h)]
end

function unscented_rule(o::Options)
    scale = NAUG + o.scaling_parameter
    Z = hcat(zeros(NAUG), sqrt(scale) * Matrix{Float64}(I, NAUG, NAUG),
        -sqrt(scale) * Matrix{Float64}(I, NAUG, NAUG))
    weights = fill(1 / (2scale), NSIGMA)
    weights[1] = o.scaling_parameter / scale
    return Z, weights
end

"Full augmented UT with 25 explicitly propagated sigma points, including W and H."
function propagation_arc(o::Options, Lc, mean, L, feedforward, gain)
    Z, w = unscented_rule(o)
    sigma_x = mean .+ L * Z[1:NX, :]
    sigma_u = feedforward .+ gain * (sigma_x .- mean)
    Y = hcat((shark_step(sigma_x[:, s], sigma_u[:, s],
        sqrt(o.dt) * Z[5:8, s], sqrt(o.dt / 12) * Z[9:12, s], Lc, o.dt)
        for s in 1:NSIGMA)...)
    predicted_mean = Y * w
    residual = Y .- predicted_mean
    P = (residual .* w') * residual'
    control_mean = sigma_u * w
    control_residual = sigma_u .- control_mean
    S = (control_residual .* w') * control_residual'
    return predicted_mean, Matrix(Symmetric((P + P') / 2)), Matrix(Symmetric((S + S') / 2))
end

function validate_shark(o::Options, Ac, Qc, Lc)
    _, P, _ = propagation_arc(o, Lc, zeros(NX), zeros(NX, NX), zeros(NU), zeros(NU, NX))
    discrepancy = maximum(abs, P - accumulated_process_covariance(Ac, Qc, o.dt))
    discrepancy < 1e-13 || error("ShARK process covariance validation failed: $discrepancy")
    return discrepancy
end

@inline control_norm(u1, u2, epsilon) = sqrt(u1^2 + u2^2 + epsilon^2)
@inline function spectral_radius_sqrt(s11, s22, s12, smoothing)
    return sqrt((s11 + s22) / 2 + sqrt((s11 - s22)^2 / 4 + s12^2 + smoothing^2))
end
