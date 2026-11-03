# Psyche checks on the exported mission arrays (included by runtests.jl when present): the
# data and uncertainty model, ballistic Mars, the analytic B-plane Jacobian, and the full
# stochastic model of both problems at a perturbed seed (exported fuel mesh, no reoptimisation) against the
# literal arc map, with directional central differences of its Jacobian and Lagrangian Hessian.

"The Psyche model from the exported mesh (no fuel reoptimisation)."
function psyche_setup(; objective = :standard, backend = nothing)
    o = C.Options(; case_id = "psyche", objective)
    d = C.load_psyche_data(C.PSYCHE_INPUT_FILE)
    u = C.psyche_uncertainty(d, o)
    params = C.psyche_phase_params(d, o, u)
    ref = (means = d.seed_states, feedforward = d.seed_controls, days = d.seed_days, steps = C.arc_steps(d, d.seed_days))
    seed, scaling = C.psyche_initial_guess(d, o, u, params, ref)
    blocks = C.psyche_stochastic_model(d, o, u, params, ref, seed, scaling, backend)
    return (; o, d, u, params, ref, seed, scaling, blocks)
end

"Symmetric sparse Lagrangian Hessian of the model from its lower-triangle coordinates."
function sparse_hessian(model, x, y)
    rows, cols = sparsity(model, false)
    H = sparse(rows, cols, Array(NLPModels.hess_coord(model, x, y)), model.meta.nvar, model.meta.nvar)
    return H + H' - Diagonal(diag(H))
end

sparse_jacobian(model, x) = (rc = sparsity(model, true);
    sparse(rc[1], rc[2], Array(NLPModels.jac_coord(model, x)), model.meta.ncon, model.meta.nvar))

const PSYCHE = psyche_setup()

@testset "Psyche data and uncertainty model" begin
    (; d, u, params, seed, scaling) = PSYCHE
    @test sqrt(u.initial_covariance_si[1, 1]) == 2.5e6
    @test u.initial_covariance_si[4:6, 4:6] == d.launch_covariance_si[4:6, 4:6]
    # Diagonal normalization by the initial standard deviations: P0 becomes its correlation matrix.
    @test maximum(abs, diag(u.initial_covariance) .- 1) < 1e-14
    @test C.psyche_si_covariance(u, u.initial_covariance) ≈ u.initial_covariance_si rtol = 1e-14
    @test length(u.navigation_covariance) == 6
    # Full node scales S_k = chol(seed P_k): every seed factor is the identity.
    identity = C.pack_lower(Matrix(1.0I, C.NX, C.NX))
    @test scaling.full && scaling.initial_factor == identity && all(==(identity), eachcol(scaling.factors))
    S1 = C.node_scale(scaling, 1)
    @test maximum(abs, S1 * S1' - u.initial_covariance) < 1e-14
    @test maximum(2:d.n+1) do k
        S = C.node_scale(scaling, k)
        maximum(abs, S * S' - seed.covariances[:, :, k]) / maximum(abs, seed.covariances[:, :, k])
    end < 1e-13
    @test u.terminal_std_si == [1.5e6, 1.5e6, 1.5e6, 2.0, 2.0, 2.0]
    @test params[2].tmax == 0 && params[2].inner == 1 && params[2].n_points == 2C.UT_DIMENSION
    @test params[1].inner == 6 && params[1] == params[3]
    @test count(>(0), d.slots) == count(!=(2), d.phase)
    @test sum(C.arc_weights(d, d.seed_days)) ≈ d.n
end

@testset "Psyche ballistic Mars" begin
    (; d, params, ref) = PSYCHE
    k = findfirst(==(2), d.phase) + 3
    theta = SVector{12}([ref.means[:, k]; 0.3; -0.2; 0.1; ref.days])
    J = ForwardDiff.jacobian(v -> C.psyche_arc_end(d, params, k, v), theta)
    @test C.psyche_arc_end(d, params, k, theta)[7] == theta[7]       # no mass flow
    @test all(iszero, J[:, 8:10])                                     # controls have no effect
    _, P, T = C.propagation_arc(params[2], ref.means[:, k], Matrix(1.0I, 7, 7), [0.3, -0.2, 0.1], ones(3, 6), 0.01)
    @test iszero(T) && P[7, 7] ≈ 1
end

@testset "Psyche B-plane Jacobian" begin
    (; d, ref) = PSYCHE
    z = SVector{6}(ref.means[1:6, d.ca_node] .* d.node_scales[1:6, d.ca_node])
    mu, rho = d.mu[2], d.mars_radius_m + C.PSYCHE_MINIMUM_MARS_ALTITUDE_M
    J = C.bplane_jacobian(z, mu, rho)
    @test J ≈ ForwardDiff.jacobian(v -> C.scaled_bplane(v, mu, rho), z) rtol = 1e-10
end

@testset "Psyche stochastic model ($objective)" for objective in (:standard, :chance_sum)
    (; o, d, u, params, seed, scaling, blocks) = objective == :standard ? PSYCHE : psyche_setup(; objective)
    model, layout, n = blocks.model, blocks.layout, d.n
    nc = count(>(0), d.slots)
    step4 = objective == :chance_sum
    @test model.meta.nvar == 7(n + 1) + 22nc + 28n + (step4 ? 21 + 7nc : 0)
    @test model.meta.ncon == 19 + 35n + 2nc + 1 + (step4 ? 21 + 6nc : 0)
    rng = Xoshiro(13)
    x = Array(model.meta.x0) .* (1 .+ 1e-4 .* randn(rng, model.meta.nvar))
    # unit directions (the rows ||d_k||^2 = 1): the cost's Gamma_k is ||S_k|| only there
    directions = reshape(view(x, blocks.ff.offset .+ (1:3nc)), 3, nc)
    directions ./= sqrt.(sum(abs2, directions; dims = 1))
    sol = C.psyche_solution(x, blocks, seed)
    values = Array(NLPModels.cons(model, x))
    offset = length(layout.boundary)
    mean_error, covariance_error = 0.0, 0.0
    for k in 1:n
        mu, P, _ = C.propagation_arc_factor(params[d.phase[k]], sol.means[:, k], C.node_factor(sol.factors, scaling, k),
            sol.feedforward[:, k], sol.gains[:, :, k], layout.steps[k])
        mean_error = max(mean_error, maximum(abs, values[offset .+ 7(k-1) .+ (1:7)] -
            (sol.means[:, k+1] - layout.ratio[:, k] .* mu - layout.defect_rhs[:, k])))
        # square-root rows L_{k+1} - chol(V_k), V_k the UT covariance in the coordinates of node k+1
        R = cholesky(Symmetric(C.node_scaled_covariance(scaling, k + 1, P))).L
        expected = sol.factors[:, k] - C.pack_lower(Matrix(R))
        covariance_error = max(covariance_error, maximum(abs, values[offset + 7n .+ 28(k-1) .+ (1:28)] - expected))
    end
    @info "Psyche oracle rows against the literal arc map" mean_error covariance_error
    @test mean_error < 1e-11
    @test covariance_error < 1e-9
    if step4
        slack = blocks.chance_slack
        gamma, tau = C.variable_values(x, blocks.throttle), C.variable_values(x, slack.tau)
        weights = layout.weights[blocks.controlled]
        @test NLPModels.obj(model, x) ≈ sum(weights .* (gamma .+ params[1].psi .* slack.scale .* tau)) rtol = 1e-13
    else
        report = C.psyche_evaluate(d, o, u, params, layout, scaling, seed, sol)
        # the model's closed-form tr(T) keeps the 3 eps of the executed covariance, the literal arc map's T does not
        jitter = sum(layout.weights[k] * o.control_covariance_weight * 3params[1].jitter for k in 1:n if d.slots[k] > 0)
        @test abs(NLPModels.obj(model, x) + blocks.objective_constant - jitter - report["objective"]) < 1e-10 * report["objective"]
    end
    @test abs(values[end] - blocks.clearance([sol.means[:, d.ca_node]; sol.factors[1:21, d.ca_node-1]])) < 1e-14

    # Directional central differences of the Jacobian and of the Lagrangian gradient.
    y = randn(rng, model.meta.ncon)
    J = sparse_jacobian(model, x)
    H = sparse_hessian(model, x, y)
    gradient(v) = Array(NLPModels.grad(model, v)) + sparse_jacobian(model, v)' * y
    for trial in 1:3
        direction = randn(rng, model.meta.nvar) .* max.(abs.(x), 1e-3)
        direction ./= maximum(abs, direction)
        h = 1e-5
        Jfd = (Array(NLPModels.cons(model, x + h * direction)) - Array(NLPModels.cons(model, x - h * direction))) / 2h
        Hfd = (gradient(x + h * direction) - gradient(x - h * direction)) / 2h
        jerr = maximum(abs, J * direction - Jfd) / max(1, maximum(abs, Jfd))
        herr = maximum(abs, H * direction - Hfd) / max(1, maximum(abs, Hfd))
        @info "Psyche directional derivative check" trial jerr herr
        @test jerr < 1e-6
        @test herr < 1e-5
    end
end
