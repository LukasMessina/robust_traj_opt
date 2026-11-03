# Initial guess of the Psyche stochastic NLP, as for the single-phase cases: reference
# means and feedforward, Bryson-rule TVLQR gains on the normalized arc Jacobians and the
# UT-propagated covariances. The TVLQR weights are scaled by the arc cost weights
# (Q_k = w_k Q, R_k = w_k R); the ballistic Mars arcs keep K = 0 but propagate the Riccati
# matrix, so cruise feedback anticipates the uncontrolled flyby. The TVLQR weights are `seed_weights`.

"""
Normalized arc Jacobians of the reference: A~ = T_end^-1 dx_end/dx T_start and
B~ = T_end^-1 dx_end/df (f = thrust / T_max), with T = diag(1 / node_scales) D.
"""
function psyche_arc_jacobians(d::PsycheData, u::PsycheUncertainty, params, ref)
    A_list, B_list = Matrix{Float64}[], Matrix{Float64}[]
    for k in 1:d.n
        theta = SVector{12}([ref.means[:, k]; ref.feedforward[:, k]; ref.days])
        J = ForwardDiff.jacobian(v -> psyche_arc_end(d, params, k, v), theta)
        start, stop = psyche_node_transform(d, u, k), psyche_node_transform(d, u, k + 1)
        push!(A_list, stop \ (Matrix(J[:, 1:7]) * start))
        push!(B_list, stop \ Matrix(J[:, 8:10]))
    end
    return A_list, B_list
end

function psyche_initial_guess(d::PsycheData, o::Options, u::PsycheUncertainty, params, ref)
    A_list, B_list = psyche_arc_jacobians(d, u, params, ref)
    controlled = d.slots .> 0
    gain_scale = [controlled[k] ? 1.0 / max(opnorm(B_list[k][1:NP, :], 2), 1e-300) : 1.0 for k in 1:d.n]
    state_weight, control_weight = seed_weights(o)
    gains = tvlqr_gains(state_weight / BRYSON_SIGMA_FACTOR^2, control_weight,
        u.terminal_transform' * u.terminal_transform ./ BRYSON_SIGMA_FACTOR^2, A_list, B_list;
        arc_weights = arc_weights(d, ref.days), controlled)
    arc_params = [params[d.phase[k]] for k in 1:d.n]
    _, covariances, controls = propagate_stochastic_moments(arc_params, ref.means, ref.feedforward, gains,
        u.initial_covariance, ref.steps)
    all(isfinite, covariances) || error("nonfinite Psyche covariance seed")
    p = params[1]
    budget = maximum(norm(ref.feedforward[:, k]) +
        p.psi * spectral_radius(SMatrix{3,3}(controls[:, :, k]), p.floor, p.smoothing_sq) for k in findall(controlled))
    log_line(@sprintf("[psyche] seed gains tvlqr (TVLQR w_P %.6g, w %.6g): max thrust budget %.3f", state_weight,
        control_weight, budget))
    G = u.terminal_transform
    slack = I - G * covariances[1:NP, 1:NP, end] * G'
    E = eigen(Symmetric((slack + slack') / 2))
    margin = cholesky(Symmetric(E.vectors * Diagonal(max.(E.values, 0.0)) * E.vectors' +
        TERMINAL_MARGIN_FLOOR^2 * I)).L
    seed = InitialGuess(copy(ref.means), copy(ref.feedforward), gains, gain_scale, pack_lower(Matrix(margin)),
        covariances, controls)
    return seed, covariance_scaling(seed, u.initial_covariance; full = true)
end
