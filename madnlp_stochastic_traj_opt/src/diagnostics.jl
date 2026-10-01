# Independent evaluation of a stochastic solution with the literal UT arc map (the
# counterpart of `recover_solution` in the JAX script). Nothing here uses the oracle
# kernels, so the reported residuals also check them.

struct StochasticSolution
    means::Matrix{Float64}             # 7 x (N+1)
    feedforward::Matrix{Float64}       # 3 x N, S_tilde
    gains::Array{Float64,3}            # 3 x 6 x N, K_tilde
    cholesky_factor::Matrix{Float64}   # 28 x N NLP variables
    terminal_margin::Vector{Float64}
    covariances::Array{Float64,3}      # 7 x 7 x (N+1), normalized P
    control_covariances::Array{Float64,3}
    radius::Vector{Float64}            # smoothed sqrt(lambda_max(T)) of the chance constraint
end

"Values of an ExaModels variable block, in its declared shape."
variable_values(x, v) = reshape(Array(x)[v.offset+1:v.offset+v.length], length.(v.size)...)

function extract_solution(x, blocks, scaling::CovarianceScaling, seed::InitialGuess, n)
    get(v) = variable_values(x, v)
    factors = get(blocks.chol)
    covariances = zeros(NX, NX, n + 1)
    covariances[:, :, 1] = node_covariance(scaling.initial_factor, scaling.std[:, 1])
    for k in 1:n
        covariances[:, :, k+1] = node_covariance(factors[:, k], scaling.std[:, k+1])
    end
    return StochasticSolution(get(blocks.means), get(blocks.ff), get(blocks.gains) .* reshape(seed.gain_scale, 1, 1, :),
        factors, get(blocks.margin), covariances, zeros(NU, NU, n), zeros(n))
end

"Literal re-evaluation of every residual and cost term; fills the control covariances and radii of `sol`."
function evaluate_solution(case::CaseData, o::Options, ref::ReferenceTraj, sol::StochasticSolution,
        scaling::CovarianceScaling, p::UTParams)
    n = n_arcs(ref)
    predicted_means = zeros(NX, n)
    predicted_covariances = zeros(NX, NX, n)
    closed_form_gap = 0.0
    for k in 1:n
        mu, P, T = propagation_arc(p, sol.means[:, k], sol.covariances[:, :, k], sol.feedforward[:, k],
            sol.gains[:, :, k], ref.steps[k])
        predicted_means[:, k] = mu
        predicted_covariances[:, :, k] = P
        sol.control_covariances[:, :, k] = T
        sol.radius[k] = spectral_radius(SMatrix{3,3}(T), p.floor, p.smoothing_sq)
        closed = control_covariance(p, SVector{3}(sol.feedforward[:, k]), SMatrix{3,6}(sol.gains[:, :, k]),
            SMatrix{6,6}(sol.covariances[1:NP, 1:NP, k]))
        closed_form_gap = max(closed_form_gap, maximum(abs, closed - T) / max(maximum(abs, T), 1e-300))
    end
    norms = [control_norm(SVector{3}(sol.feedforward[:, k]), p.eps) for k in 1:n]
    chance = norms .+ p.psi .* sol.radius
    inv_next = scaling.inv_std[:, 2:n+1]
    scaled_defects = map(1:n) do k
        L = unpack_lower(sol.cholesky_factor[:, k], NX)
        maximum(abs, L * L' - predicted_covariances[:, :, k] .* (inv_next[:, k] * inv_next[:, k]'))
    end
    inverse_target = terminal_inverse_std(o)
    target = sol.covariances[1:NP, 1:NP, end] .* (inverse_target * inverse_target')
    G = unpack_lower(sol.terminal_margin, NP)
    terminal_residual = maximum(abs, I - target - G * G')
    mean_target = terminal_mean_target(case, o, ref)
    boundary = max(maximum(abs, sol.means[:, 1] - case.x0), maximum(abs, sol.means[1:NP, end] - mean_target))
    mean_defect = maximum(abs, sol.means[:, 2:end] - predicted_means)
    equality = max(boundary, mean_defect, maximum(scaled_defects), terminal_residual)
    chance_violation = max(0.0, maximum(chance) - 1)
    bounds = max(0.0, -minimum(sol.cholesky_factor[collect(LTRI_DIAGONAL), :]),
        o.terminal_margin_floor - minimum(sol.terminal_margin[collect(MTRI_DIAGONAL)]))
    state_term = [tr(sol.covariances[1:NP, 1:NP, k]) * p.inv_sigma_sq for k in 1:n]
    control_term = [tr(sol.control_covariances[:, :, k]) for k in 1:n]
    ratio = eigvals(Symmetric(target))
    return Dict{String,Any}(
        "objective" => sum(norms) + sum(state_term) + sum(control_term),
        "objective_mean_control" => sum(ref.steps .* norms),
        "objective_state_covariance" => sum(ref.steps .* state_term),
        "objective_control_covariance" => sum(ref.steps .* control_term),
        "max_equality_residual" => equality,
        "max_control_chance_violation" => chance_violation,
        "max_bound_violation" => bounds,
        "max_terminal_constraint_violation" => terminal_residual,
        "state_matching_defect_nd" => mean_defect,
        "state_covariance_matching_defect" => maximum(abs, sol.covariances[:, :, 2:end] - predicted_covariances),
        "scaled_state_covariance_matching_defect" => maximum(scaled_defects),
        "closed_form_control_covariance_relative_gap" => closed_form_gap,
        "terminal_componentwise_mean_error_nd" => maximum(abs, sol.means[1:NP, end] - mean_target),
        "terminal_covariance_max_eigenvalue" => maximum(ratio),
        "max_thrust_budget" => maximum(chance),
        "absolute_feasibility" => max(equality, chance_violation, bounds))
end
