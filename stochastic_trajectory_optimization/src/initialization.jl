# Initial guess: energy-optimal means and feedforward, Bryson-rule TVLQR gains and
# UT-propagated covariances (JAX seed verification and the covariance seed of
# `TrajectoryNLP.__init__`).

"Normalized open-loop Jacobians of the Dopri8 arc map: A~ = D^-1 A D, B~ = D^-1 B T_max."
function normalized_arc_jacobians(case::CaseData, n::Normalization, states, controls, steps)
    # Same arithmetic as the reference: multiply by diag(1/scale), not divide by scale.
    D = Diagonal(Vector(n.scale))
    D_inv = inv(D)
    A_list, B_list = Matrix{Float64}[], Matrix{Float64}[]
    for k in eachindex(steps)
        J = ForwardDiff.jacobian([states[:, k]; controls[:, k]]) do v
            u = SVector{3}(v[8], v[9], v[10])
            Vector(deterministic_step(SVector{7}(v[1:7]), u, norm(u), steps[k], case.gravity, case.ve))
        end
        push!(A_list, D_inv * J[:, 1:7] * D)
        push!(B_list, D_inv * J[:, 8:10] * case.tmax)
    end
    return A_list, B_list
end

"""
Backward Riccati recursion on the 6-state position/velocity subsystem (mass is not observed).
Arc k uses Q_k = w_k Q and R_k = w_k R (`arc_weights`, default 1); an uncontrolled arc
(`controlled[k] == false`) has K_k = 0 and still propagates the cost-to-go.
"""
function tvlqr_gains(state_weight, control_weight, terminal_weights, A_list, B_list;
        arc_weights = nothing, controlled = nothing)
    Q = state_weight * Matrix{Float64}(I, NP, NP)
    R = control_weight * Matrix{Float64}(I, NU, NU)
    P = terminal_weights isa AbstractMatrix ? Matrix{Float64}(terminal_weights) : Matrix(Diagonal(terminal_weights))
    gains = zeros(NU, NP, length(A_list))
    for k in reverse(eachindex(A_list))
        A = A_list[k][1:NP, 1:NP]
        B = B_list[k][1:NP, :]
        Qk, Rk = arc_weights === nothing ? (Q, R) : (arc_weights[k] * Q, arc_weights[k] * R)
        if controlled === nothing || controlled[k]
            gain = (Rk + B' * P * B) \ (B' * P * A)
            gains[:, :, k] = -gain
            F = A - B * gain
            P = Qk + gain' * Rk * gain + F' * P * F
        else
            P = Qk + A' * P * A
        end
        P = 0.5 * (P + P')
    end
    return gains
end

"""
Bryson weights Q~ = w_P I/c^2, R~ = w I (`seed_weights`: the cost's tracking weights, or Bryson's rule in
step 4), Qf~ = diag(r_p I3, r_v I3)/c^2.
"""
function seed_gains(o::Options, A_list, B_list; arc_weights = nothing)
    terminal = [fill(o.position_covariance_reduction, 3); fill(o.velocity_covariance_reduction, 3)] ./ BRYSON_SIGMA_FACTOR^2
    state_weight, control_weight = seed_weights(o)
    return tvlqr_gains(state_weight / BRYSON_SIGMA_FACTOR^2, control_weight, terminal, A_list, B_list; arc_weights)
end

struct InitialGuess
    means::Matrix{Float64}          # 7 x (N+1)
    feedforward::Matrix{Float64}    # 3 x N, normalized by T_max
    gains::Array{Float64,3}         # 3 x 6 x N, K_tilde
    gain_scale::Vector{Float64}     # 1 / ||B~_k[1:6, :]||_2
    terminal_margin::Vector{Float64}
    covariances::Array{Float64,3}   # 7 x 7 x (N+1), normalized, propagated
    control_covariances::Array{Float64,3}
end

function build_initial_guess(case::CaseData, o::Options, n::Normalization, p::UTParams, ref::ReferenceTraj)
    means = copy(ref.states)
    means[:, 1] = case.x0
    feedforward = ref.controls ./ case.tmax
    A_list, B_list = normalized_arc_jacobians(case, n, ref.states, ref.controls, ref.steps)
    # Position/velocity rows only: the gains never act on mass, and the mass row
    # (normalised by the mass scale) would otherwise dominate the norm by ~1e3.
    gain_scale = [1.0 / max(opnorm(B[1:NP, :], 2), 1e-300) for B in B_list]
    # Q_k = w_k Q, R_k = w_k R with the cost's arc weights
    gains = seed_gains(o, A_list, B_list;
        arc_weights = single_phase_arc_weights(ref))
    _, covariances, controls = propagate_stochastic_moments(p, means, feedforward, gains,
        initial_covariance(n), ref.steps)
    inverse_std = terminal_inverse_std(o)
    slack = I - (inverse_std * inverse_std') .* covariances[1:NP, 1:NP, end]
    E = eigen(Symmetric((slack + slack') / 2))
    margin = cholesky(Symmetric(E.vectors * Diagonal(max.(E.values, 0.0)) * E.vectors' +
        TERMINAL_MARGIN_FLOOR^2 * I)).L
    return InitialGuess(means, feedforward, gains, gain_scale, pack_lower(Matrix(margin)),
        covariances, controls)
end

"""
Fixed node scales of the Cholesky variables. Diagonal (default): std_k = sqrt(max(diag Sigma_k,
1e-12)) of the seed covariance, and the NLP factor of node k is chol(Sigma_k) scaled by 1/std_k.
Full (`full = true`): the lower Cholesky factor S_k of the seed covariance itself,
Sigma_k = S_k L_k L_k' S_k', so every seed factor L_k is the identity; products of lower
triangular matrices stay lower triangular, so L_k remains a Cholesky factor. `std` is then
only reported.
"""
struct CovarianceScaling
    variances::Matrix{Float64}      # 7 x (N+1)
    std::Matrix{Float64}
    inv_std::Matrix{Float64}
    factors::Matrix{Float64}        # 28 x N seeds of nodes 1..N
    initial_factor::Vector{Float64} # 28, node 0 (data)
    scale::Array{Float64,3}         # 7 x 7 x (N+1) lower node scales S_k (full scaling; empty otherwise)
    full::Bool
end

CovarianceScaling(variances, std, inv_std, factors, initial_factor) =
    CovarianceScaling(variances, std, inv_std, factors, initial_factor, zeros(NX, NX, 0), false)

function covariance_scaling(seed::InitialGuess, P0::AbstractMatrix; full::Bool = false)
    C = seed.covariances
    N = size(C, 3) - 1
    variances = max.(reduce(hcat, [diag(C[:, :, k]) for k in 1:N+1]), 1e-12)
    std = sqrt.(variances)
    inv_std = 1.0 ./ std
    if full
        scale = zeros(NX, NX, N + 1)
        scale[:, :, 1] = cholesky(Symmetric(P0)).L
        for k in 2:N+1
            scale[:, :, k] = cholesky(Symmetric((C[:, :, k] + C[:, :, k]') / 2)).L
        end
        identity = pack_lower(Matrix(1.0I, NX, NX))
        return CovarianceScaling(variances, std, inv_std, repeat(identity, 1, N), identity, scale, true)
    end
    factors = zeros(NL, N)
    for k in 1:N
        scaled = C[:, :, k+1] .* (inv_std[:, k+1] * inv_std[:, k+1]')
        factors[:, k] = pack_lower(Matrix(cholesky(Symmetric(scaled)).L))
    end
    initial = Matrix(cholesky(Symmetric(P0)).L) .* inv_std[:, 1]
    return CovarianceScaling(variances, std, inv_std, factors, pack_lower(initial))
end

"Node scale of node k as a dense lower matrix: S_k (full scaling) or diag(std_k)."
node_scale(scaling::CovarianceScaling, k) = scaling.full ? scaling.scale[:, :, k] : Matrix(Diagonal(scaling.std[:, k]))

"Normalized covariance of node k from the factor variable: (S_k L)(S_k L)' (full) or (std .* L)(std .* L)'."
function scaled_factor(scaling::CovarianceScaling, k, L::AbstractMatrix)
    return scaling.full ? scaling.scale[:, :, k] * L : L .* scaling.std[:, k]
end

"Covariance in the scaled coordinates of node k: S_k^-1 P S_k^-T (full) or P ./ (std std')."
function node_scaled_covariance(scaling::CovarianceScaling, k, P::AbstractMatrix)
    if scaling.full
        S = LowerTriangular(scaling.scale[:, :, k])
        return S \ Matrix(P) / S'
    end
    return P .* (scaling.inv_std[:, k] * scaling.inv_std[:, k]')
end

"Packed lower node scales (28 x m) and packed inverses of the nodes `nodes`, for the oracles."
packed_node_scales(scaling::CovarianceScaling, nodes) =
    reduce(hcat, [pack_lower(node_scale(scaling, k)) for k in nodes])
packed_inverse_node_scales(scaling::CovarianceScaling, nodes) =
    reduce(hcat, [pack_lower(Matrix(inv(LowerTriangular(node_scale(scaling, k))))) for k in nodes])

covariance_scaling(seed::InitialGuess, n::Normalization) = covariance_scaling(seed, initial_covariance(n))
