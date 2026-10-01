# Initial guess: energy-optimal means and feedforward, Bryson-rule TVLQR gains and
# UT-propagated covariances (JAX `build_initial_guess` and the covariance seed of
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
            Vector(deterministic_step(SVector{7}(v[1:7]), u, norm(u), steps[k], case.mu, case.ve))
        end
        push!(A_list, D_inv * J[:, 1:7] * D)
        push!(B_list, D_inv * J[:, 8:10] * case.tmax)
    end
    return A_list, B_list
end

"Backward Riccati recursion on the 6-state position/velocity subsystem (mass is not observed)."
function tvlqr_gains(state_weight, control_weight, terminal_weights, A_list, B_list)
    Q = state_weight * Matrix{Float64}(I, NP, NP)
    R = control_weight * Matrix{Float64}(I, NU, NU)
    P = Matrix(Diagonal(terminal_weights))
    gains = zeros(NU, NP, length(A_list))
    for k in reverse(eachindex(A_list))
        A = A_list[k][1:NP, 1:NP]
        B = B_list[k][1:NP, :]
        gain = (R + B' * P * B) \ (B' * P * A)
        gains[:, :, k] = -gain
        F = A - B * gain
        P = Q + gain' * R * gain + F' * P * F
        P = 0.5 * (P + P')
    end
    return gains
end

"Bryson weights Q~ = I/c^2, R~ = I, Qf~ = diag(r_p I3, r_v I3)/c^2."
function seed_gains(o::Options, A_list, B_list)
    state_weight = 1.0 / o.bryson_sigma_factor^2
    terminal = state_weight .* [fill(o.position_covariance_reduction, 3); fill(o.velocity_covariance_reduction, 3)]
    return tvlqr_gains(state_weight, 1.0, terminal, A_list, B_list)
end

struct InitialGuess
    means::Matrix{Float64}          # 7 x (N+1)
    feedforward::Matrix{Float64}    # 3 x N, normalized by T_max
    gains::Array{Float64,3}         # 3 x 6 x N, K_tilde
    gain_scale::Vector{Float64}     # 1 / ||B~_k||_2
    terminal_margin::Vector{Float64}
    covariances::Array{Float64,3}   # 7 x 7 x (N+1), normalized, propagated
    control_covariances::Array{Float64,3}
end

function build_initial_guess(case::CaseData, o::Options, n::Normalization, p::UTParams, ref::ReferenceTraj)
    means = copy(ref.states)
    means[:, 1] = case.x0
    feedforward = ref.controls ./ case.tmax
    A_list, B_list = normalized_arc_jacobians(case, n, ref.states, ref.controls, ref.steps)
    gain_scale = [1.0 / max(opnorm(B, 2), 1e-300) for B in B_list]
    gains = seed_gains(o, A_list, B_list)
    _, covariances, controls = propagate_stochastic_moments(p, means, feedforward, gains,
        initial_covariance(n), ref.steps)
    inverse_std = terminal_inverse_std(o)
    slack = I - (inverse_std * inverse_std') .* covariances[1:NP, 1:NP, end]
    E = eigen(Symmetric((slack + slack') / 2))
    margin = cholesky(Symmetric(E.vectors * Diagonal(max.(E.values, 0.0)) * E.vectors' +
        o.terminal_margin_floor^2 * I)).L
    return InitialGuess(means, feedforward, gains, gain_scale, pack_lower(Matrix(margin)),
        covariances, controls)
end

"""
Fixed node scales of the Cholesky variables: std_k = sqrt(max(diag Sigma_k, 1e-12))
of the seed covariance; the NLP factor of node k is chol(Sigma_k) scaled by 1/std_k.
"""
struct CovarianceScaling
    variances::Matrix{Float64}      # 7 x (N+1)
    std::Matrix{Float64}
    inv_std::Matrix{Float64}
    factors::Matrix{Float64}        # 28 x N seeds of nodes 1..N
    initial_factor::Vector{Float64} # 28, node 0 (data)
end

function covariance_scaling(seed::InitialGuess, n::Normalization)
    C = seed.covariances
    N = size(C, 3) - 1
    variances = max.(reduce(hcat, [diag(C[:, :, k]) for k in 1:N+1]), 1e-12)
    std = sqrt.(variances)
    inv_std = 1.0 ./ std
    factors = zeros(NL, N)
    for k in 1:N
        scaled = C[:, :, k+1] .* (inv_std[:, k+1] * inv_std[:, k+1]')
        factors[:, k] = pack_lower(Matrix(cholesky(Symmetric(scaled)).L))
    end
    initial = Matrix(cholesky(Symmetric(initial_covariance(n))).L) .* inv_std[:, 1]
    return CovarianceScaling(variances, std, inv_std, factors, pack_lower(initial))
end
