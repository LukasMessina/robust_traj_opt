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

"Feedforward S_k of the controlled arcs: the ff variables, or Gamma_k d_k of the polar and throttle controls."
function feedforward_values(x, blocks)
    theta = variable_values(x, blocks.ff)
    gamma = get(blocks, :throttle, nothing)
    gamma === nothing || return theta .* reshape(variable_values(x, gamma), 1, :)
    D = get(blocks, :polar_directions, nothing)
    D === nothing && return theta
    return reduce(hcat, [Vector(theta[1, k] * polar_direction(theta[2, k], theta[3, k], SVector{3}(D[:, k])))
                         for k in axes(theta, 2)])
end

function extract_solution(x, blocks, scaling::CovarianceScaling, seed::InitialGuess, n)
    get(v) = variable_values(x, v)
    factors = get(blocks.chol)
    covariances = zeros(NX, NX, n + 1)
    covariances[:, :, 1] = node_covariance(scaling.initial_factor, scaling.std[:, 1])
    for k in 1:n
        covariances[:, :, k+1] = node_covariance(factors[:, k], scaling.std[:, k+1])
    end
    return StochasticSolution(get(blocks.means), feedforward_values(x, blocks), get(blocks.gains) .* reshape(seed.gain_scale, 1, 1, :),
        factors, blocks.margin === nothing ? zeros(NM) : get(blocks.margin), covariances, zeros(NU, NU, n), zeros(n))
end

"Lower factor F (P = F F') of node k: the NLP factor of nodes 2..N+1 or the fixed initial factor, times the node scales."
node_factor(factors, scaling::CovarianceScaling, k) =
    scaled_factor(scaling, k, unpack_lower(k == 1 ? scaling.initial_factor : factors[:, k-1], NX))

"Literal re-evaluation of every residual and cost term; fills the control covariances and radii of `sol`."
function evaluate_solution(case::CaseData, o::Options, ref::ReferenceTraj, sol::StochasticSolution,
        scaling::CovarianceScaling, p::UTParams)
    n = n_arcs(ref)
    predicted_means = zeros(NX, n)
    predicted_covariances = zeros(NX, NX, n)
    closed_form_gap = 0.0
    for k in 1:n
        mu, P, T = propagation_arc_factor(p, sol.means[:, k], node_factor(sol.cholesky_factor, scaling, k),
            sol.feedforward[:, k], sol.gains[:, :, k], ref.steps[k])
        predicted_means[:, k] = mu
        predicted_covariances[:, :, k] = P
        sol.control_covariances[:, :, k] = T
        sol.radius[k] = spectral_radius(SMatrix{3,3}(T), p.floor, p.smoothing_sq)
        closed = control_covariance(p, SVector{3}(sol.feedforward[:, k]), SMatrix{3,6}(sol.gains[:, :, k]),
            SMatrix{6,6}(sol.covariances[1:NP, 1:NP, k]))
        closed_form_gap = max(closed_form_gap, maximum(abs, closed - T) / max(maximum(abs, T), 1e-300))
    end
    norms = [feedforward_norm(p, SVector{3}(sol.feedforward[:, k])) for k in 1:n]
    chance = norms .+ p.psi .* sol.radius
    inv_next = scaling.inv_std[:, 2:n+1]
    scaled_defects = map(1:n) do k
        L = unpack_lower(sol.cholesky_factor[:, k], NX)
        maximum(abs, L * L' - predicted_covariances[:, :, k] .* (inv_next[:, k] * inv_next[:, k]'))
    end
    inverse_target = terminal_inverse_std(o)
    target = sol.covariances[1:NP, 1:NP, end] .* (inverse_target * inverse_target')
    G = unpack_lower(sol.terminal_margin, NP)
    terminal_bound = o.terminal_covariance == :constraint
    # with the terminal covariance in the cost there is no margin: the violation of G P_N G' <= I is reported instead
    terminal_residual = terminal_bound ? maximum(abs, I - target - G * G') : max(0.0, eigmax(Symmetric(target)) - 1)
    mean_target = terminal_mean_target(case, o, ref)
    boundary = max(maximum(abs, sol.means[:, 1] - case.x0), maximum(abs, sol.means[1:NP, end] - mean_target))
    mean_defect = maximum(abs, sol.means[:, 2:end] - predicted_means)
    equality = max(boundary, mean_defect, maximum(scaled_defects), terminal_residual)
    chance_violation = max(0.0, maximum(chance) - 1)
    bounds = max(0.0, -minimum(sol.cholesky_factor[collect(LTRI_DIAGONAL), :]),
        terminal_bound ? o.terminal_margin_floor - minimum(sol.terminal_margin[collect(MTRI_DIAGONAL)]) : 0.0)
    state_term = [tr(sol.covariances[1:NP, 1:NP, k]) / o.bryson_sigma_factor^2 for k in 1:n]
    control_term = [tr(sol.control_covariances[:, :, k]) for k in 1:n]
    # Shares of the standard cost (tracking and arc weights applied), whatever the objective of the run.
    w = single_phase_arc_weights(o, ref)
    terminal_cost = terminal_bound ? 0.0 : o.terminal_covariance_weight * tr(target) / o.bryson_sigma_factor^2
    weighted = (sum(w .* norms), o.state_covariance_weight * sum(w .* state_term),
        o.control_covariance_weight * sum(w .* control_term), terminal_cost)
    ratio = eigvals(Symmetric(target))
    return Dict{String,Any}(
        "control_chance_covariance" => "executed control; commanded dispersion, navigation and execution included",
        # weighted like the NLP cost; the components below are the physical (unweighted) terms
        "objective" => o.objective == :quantile ? sum(chance) : o.objective == :chance_sum ? sum(w .* chance) :
            sum(weighted),
        "objective_kind" => String(o.objective),
        "objective_mean_control" => sum(ref.steps .* norms),
        "objective_state_covariance" => sum(ref.steps .* state_term),
        "objective_control_covariance" => sum(ref.steps .* control_term),
        "objective_share_mean_control" => weighted[1] / sum(weighted),
        "objective_share_state_covariance" => weighted[2] / sum(weighted),
        "objective_share_control_covariance" => weighted[3] / sum(weighted),
        "objective_terminal_covariance" => terminal_cost,
        "objective_share_terminal_covariance" => weighted[4] / sum(weighted),
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
