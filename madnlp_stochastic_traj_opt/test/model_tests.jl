# Whole-model checks on a truncated problem (3 arcs of the reference mesh, CPU, CR3BP and heliocentric), for
# both problems: constraint values against the literal UT arc map, the objective against the solution report
# (step 3) or the epigraph sum (step 4), the Jacobian against central differences of the constraints, and the
# Lagrangian Hessian (native algebra and every oracle) against central differences of its exact gradient.

@testset "stochastic model ($case_id, $objective)" for case_id in ("lyapunov_l1_to_l2", "earth_mars"),
        objective in (:standard, :chance_sum)
    s = truncated_setup(; case_id, objective)
    model, n = s.blocks.model, C.n_arcs(s.ref)
    step4 = objective == :chance_sum
    @test model.meta.nvar == 57n + 7 + (step4 ? 7n + 21 : 0)
    @test model.meta.ncon == 37n + 13 + (step4 ? 6n + 21 : 0)
    rng = Xoshiro(3)
    x = Array(model.meta.x0) .* (1 .+ 1e-3 .* randn(rng, model.meta.nvar))
    # unit directions (the rows ||d_k||^2 = 1): the cost's Gamma_k is ||S_k|| only there
    directions = reshape(view(x, s.blocks.ff.offset .+ (1:3n)), 3, n)
    directions ./= sqrt.(sum(abs2, directions; dims = 1))

    sol = C.extract_solution(x, s.blocks, s.scaling, s.seed, n)
    values = Array(NLPModels.cons(model, x))
    offset = C.NX + C.NP   # boundary rows precede the defects
    for k in 1:n
        mu, P, _ = C.propagation_arc_factor(s.p, sol.means[:, k], C.node_factor(sol.cholesky_factor, s.scaling, k),
            sol.feedforward[:, k], sol.gains[:, :, k], s.ref.steps[k])
        @test maximum(abs, values[offset .+ 7(k-1) .+ (1:7)] - (sol.means[:, k+1] - mu)) < 1e-12
        # square-root rows L_{k+1} - chol(V_k), V_k the UT covariance in the coordinates of node k+1
        inv = s.scaling.inv_std[:, k+1]
        R = cholesky(Symmetric(P .* (inv * inv'))).L
        @test maximum(abs, values[offset + 7n .+ 28(k-1) .+ (1:28)] - (sol.cholesky_factor[:, k] - C.pack_lower(Matrix(R)))) < 1e-10
    end

    report = C.evaluate_solution(s.case, s.o, s.ref, sol, s.scaling, s.p)
    # the closed form keeps the 1e-12 I of the executed covariance, the jitter-free Gates factors do not: a gap of
    # 1e-12 / max |T| (max |T| >= 1e-4 here)
    @test report["closed_form_control_covariance_relative_gap"] < 1e-8
    if step4
        slack = s.blocks.chance_slack
        gamma, tau = C.variable_values(x, s.blocks.throttle), C.variable_values(x, slack.tau)
        weights = C.single_phase_arc_weights(s.ref)
        @test NLPModels.obj(model, x) ≈ sum(weights .* (gamma .+ s.p.psi .* slack.scale .* tau)) rtol = 1e-13
    else
        @test abs(NLPModels.obj(model, x) + s.blocks.objective_constant - report["objective"]) < 1e-10 * report["objective"]
    end

    J = dense_jacobian(model, x)
    @test maximum(abs, J - central_differences(v -> Array(NLPModels.cons(model, v)), x)) < 1e-6 * max(1, maximum(abs, J))

    y = randn(rng, model.meta.ncon)
    H = dense_hessian(model, x, y)
    Hfd = central_differences(v -> lagrangian_gradient(model, v, y), x)
    err = maximum(abs, H - (Hfd + Hfd') / 2) / max(1, maximum(abs, H))
    @info "relative Hessian error vs central differences ($case_id, $objective)" err
    @test err < 1e-5
end

# Step 4: the chance rows Gamma_k + psi s_k tau_hat_k <= 1 and the 6N slack rows against an independent evaluation
# of `control_covariance`, at the start point (where the slack rows vanish by construction) and at a perturbed point.
@testset "slack epigraph rows ($case_id)" for case_id in ("lyapunov_l1_to_l2", "earth_mars")
    s = truncated_setup(; case_id, objective = :chance_sum)
    model, n, slack = s.blocks.model, C.n_arcs(s.ref), s.blocks.chance_slack
    # native rows in creation order (boundary, defects, terminal, chance, unit directions), then the oracle rows
    terminal_rows = 35n + 13 .+ (1:C.NM)
    chance_rows = terminal_rows[end] .+ (1:n)
    slack_rows = model.meta.ncon - 6n .+ (1:6n)
    @test maximum(abs, Array(NLPModels.cons(model, Array(model.meta.x0)))[slack_rows]) < 1e-12

    rng = Xoshiro(5)
    x = Array(model.meta.x0) .* (1 .+ 1e-3 .* randn(rng, model.meta.nvar))
    values = Array(NLPModels.cons(model, x))
    sol = C.extract_solution(x, s.blocks, s.scaling, s.seed, n)
    tau, G = C.variable_values(x, slack.tau), C.variable_values(x, slack.factor)
    gamma = C.variable_values(x, s.blocks.throttle)
    for a in 1:n
        f = SVector{3}(sol.feedforward[:, a])
        T = C.control_covariance(s.p, f, SMatrix{3,6}(sol.gains[:, :, a]), SMatrix{6,6}(sol.covariances[1:6, 1:6, a]))
        Gm = C.slack_matrix(SVector{6}(G[:, a]))
        expected = [tau[a]^2 * (i == j) - (Gm * Gm')[i, j] - T[i, j] / slack.scale[a]^2 for (i, j) in zip(C.SLACK_ROWS, C.SLACK_COLS)]
        @test maximum(abs, values[slack_rows[6(a-1) .+ (1:6)]] - expected) < 1e-10
        @test abs(values[chance_rows[a]] - (gamma[a] + s.p.psi * slack.scale[a] * tau[a])) < 1e-14
    end
end

@testset "execution uncertainty enters the thrust chance constraints" begin
    s = truncated_setup(; arcs = 1, case_id = "earth_mars")
    f = SVector(0.2, -0.1, 0.05)
    K = zero(SMatrix{3,6,Float64})
    P = SMatrix{6,6}(Matrix(1.0I, 6, 6))
    @test iszero(C.commanded_control_covariance(s.p, K, P))
    @test tr(C.control_covariance(s.p, f, K, P)) > 0
    @test C.control_spread(s.p, f, K, P) == s.p.psi * C.spectral_radius(C.control_covariance(s.p, f, K, P), s.p.floor, s.p.smoothing_sq)
    @test C.control_spread(s.p, f, K, P) > s.p.psi * C.spectral_radius(zero(SMatrix{3,3,Float64}), s.p.floor, s.p.smoothing_sq)
end
