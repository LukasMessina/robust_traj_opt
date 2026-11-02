# Whole-model checks on a truncated problem (3 arcs, CPU, CR3BP and heliocentric): constraint values
# against the literal UT arc map, the objective against the literal nested-UT
# costs, the Jacobian against central differences of the constraints, and the
# Lagrangian Hessian (native algebra + both oracles) against central differences
# of the exact Lagrangian gradient.

@testset "stochastic model derivatives ($case_id)" for case_id in ("lyapunov_l1_to_l2", "earth_mars")
    s = truncated_setup(; case_id)
    model, n = s.blocks.model, C.n_arcs(s.ref)
    @test model.meta.nvar == 56n + 28
    @test model.meta.ncon == 36n + 34
    rng = Xoshiro(3)
    x = Array(model.meta.x0) .* (1 .+ 1e-3 .* randn(rng, model.meta.nvar))

    sol = C.extract_solution(x, s.blocks, s.scaling, s.seed, n)
    values = Array(NLPModels.cons(model, x))
    offset = C.NX + C.NP   # boundary rows precede the defects
    for k in 1:n
        mu, P, _ = C.propagation_arc_factor(s.p, sol.means[:, k], C.node_factor(sol.cholesky_factor, s.scaling, k),
            sol.feedforward[:, k], sol.gains[:, :, k], s.ref.steps[k])
        @test maximum(abs, values[offset .+ 7(k-1) .+ (1:7)] - (sol.means[:, k+1] - mu)) < 1e-12
        inv = s.scaling.inv_std[:, k+1]
        L = C.unpack_lower(sol.cholesky_factor[:, k], C.NX)
        expected = C.pack_lower(L * L' - P .* (inv * inv'))
        @test maximum(abs, values[offset + 7n .+ 28(k-1) .+ (1:28)] - expected) < 1e-10
    end

    report = C.evaluate_solution(s.case, s.o, s.ref, sol, s.scaling, s.p)
    @test abs(NLPModels.obj(model, x) + s.blocks.objective_constant - report["objective"]) < 1e-10 * report["objective"]
    @test report["closed_form_control_covariance_relative_gap"] < 1e-12

    J = dense_jacobian(model, x)
    @test maximum(abs, J - central_differences(v -> Array(NLPModels.cons(model, v)), x)) < 1e-6 * max(1, maximum(abs, J))

    y = randn(rng, model.meta.ncon)
    H = dense_hessian(model, x, y)
    Hfd = central_differences(v -> lagrangian_gradient(model, v, y), x)
    err = maximum(abs, H - (Hfd + Hfd') / 2) / max(1, maximum(abs, H))
    @info "relative Hessian error vs central differences" err
    @test err < 1e-5
end

# chance_formulation = :slack: the chance rows and the 6N slack rows against an independent evaluation of
# `control_covariance`, at the start point (where the slack rows vanish by construction) and at a perturbed
# point, plus exact first and second derivatives against central differences.
@testset "slack chance constraint ($case_id)" for case_id in ("lyapunov_l1_to_l2", "earth_mars")
    s = truncated_setup(; case_id, chance_formulation = :slack)
    model, n, slack = s.blocks.model, C.n_arcs(s.ref), s.blocks.chance_slack
    @test model.meta.nvar == 63n + 28
    @test model.meta.ncon == 42n + 34
    chance_rows, slack_rows = 35n + 34 .+ (1:n), 36n + 34 .+ (1:6n)
    @test maximum(abs, Array(NLPModels.cons(model, Array(model.meta.x0)))[slack_rows]) < 1e-12

    rng = Xoshiro(5)
    x = Array(model.meta.x0) .* (1 .+ 1e-3 .* randn(rng, model.meta.nvar))
    values = Array(NLPModels.cons(model, x))
    sol = C.extract_solution(x, s.blocks, s.scaling, s.seed, n)
    tau, G = C.variable_values(x, slack.tau), C.variable_values(x, slack.factor)
    for a in 1:n
        f = SVector{3}(sol.feedforward[:, a])
        T = C.control_covariance(s.p, f, SMatrix{3,6}(sol.gains[:, :, a]), SMatrix{6,6}(sol.covariances[1:6, 1:6, a]))
        Gm = C.lower_matrix3(SVector{6}(G[:, a]))
        expected = C.pack_lower(Matrix(tau[a]^2 * I - T / slack.scale[a]^2 - Gm * Gm'))
        @test maximum(abs, values[slack_rows[6(a-1) .+ (1:6)]] - expected) < 1e-10
        @test abs(values[chance_rows[a]] - (C.control_norm(f, s.p.eps) + s.p.psi * slack.scale[a] * tau[a])) < 1e-14
    end

    J = dense_jacobian(model, x)
    @test maximum(abs, J - central_differences(v -> Array(NLPModels.cons(model, v)), x)) < 1e-6 * max(1, maximum(abs, J))
    y = randn(rng, model.meta.ncon)
    H = dense_hessian(model, x, y)
    Hfd = central_differences(v -> lagrangian_gradient(model, v, y), x)
    @test maximum(abs, H - (Hfd + Hfd') / 2) / max(1, maximum(abs, H)) < 1e-5
end

@testset "execution uncertainty enters thrust chance constraints" begin
    s = truncated_setup(; arcs=1, case_id="earth_mars")
    K = zero(SMatrix{3,6,Float64})
    P = SMatrix{6,6}(s.seed.covariances[1:6,1:6,1])
    f = SVector(.3,.2,.1)
    @test iszero(C.commanded_control_covariance(s.p,K,P))
    @test tr(C.control_covariance(s.p,f,K,P)) > 0
    @test C.control_chance(s.p,f,K,P) == C.control_norm(f,s.p.eps) +
        s.p.psi*C.spectral_radius(C.control_covariance(s.p,f,K,P),s.p.floor,s.p.smoothing_sq)
    @test C.control_chance(s.p,f,K,P) > C.control_norm(f,s.p.eps) +
        s.p.psi*C.spectral_radius(zero(SMatrix{3,3,Float64}),s.p.floor,s.p.smoothing_sq)
end

# objective = :quantile: epigraph variables q_k <= 1, chance rows ||S_k||_eps + psi rho(T_k) - q_k <= 0
# and objective sum_k q_k. Rows against the literal chance values, exact first and second
# derivatives against central differences.
@testset "quantile objective ($case_id)" for case_id in ("lyapunov_l1_to_l2", "earth_mars")
    s = truncated_setup(; case_id, objective = :quantile)
    model, n = s.blocks.model, C.n_arcs(s.ref)
    @test model.meta.nvar == 57n + 28
    @test model.meta.ncon == 36n + 34
    @test s.blocks.objective_constant == 0
    chance_rows, q = 35n + 34 .+ (1:n), 56n + 28 .+ (1:n)
    @test all(model.meta.uvar[q] .== 1)
    x0 = Array(model.meta.x0)
    @test maximum(abs, Array(NLPModels.cons(model, x0))[chance_rows]) < 1e-12   # q starts at the seed quantile

    rng = Xoshiro(7)
    x = x0 .* (1 .+ 1e-3 .* randn(rng, model.meta.nvar))
    sol = C.extract_solution(x, s.blocks, s.scaling, s.seed, n)
    report = C.evaluate_solution(s.case, s.o, s.ref, sol, s.scaling, s.p)
    chance = [C.control_norm(SVector{3}(sol.feedforward[:, k]), s.p.eps) for k in 1:n] .+ s.p.psi .* sol.radius
    @test maximum(abs, Array(NLPModels.cons(model, x))[chance_rows] - (chance - x[q])) < 1e-12
    @test abs(NLPModels.obj(model, x) - sum(x[q])) < 1e-14 * sum(x[q])
    @test abs(report["objective"] - sum(chance)) < 1e-14 * sum(chance)

    J = dense_jacobian(model, x)
    @test maximum(abs, J - central_differences(v -> Array(NLPModels.cons(model, v)), x)) < 1e-6 * max(1, maximum(abs, J))
    @test all(J[chance_rows, q] .== -Matrix(I, n, n))
    y = randn(rng, model.meta.ncon)
    H = dense_hessian(model, x, y)
    Hfd = central_differences(v -> lagrangian_gradient(model, v, y), x)
    @test maximum(abs, H - (Hfd + Hfd') / 2) / max(1, maximum(abs, H)) < 1e-5
end

# objective = :quantile with chance_formulation = :slack: the cost is the sum of the slack chance
# rows' left-hand sides ||S_k||_eps + psi s_k tau_hat_k (no epigraph variables).
@testset "quantile objective, slack form ($case_id)" for case_id in ("lyapunov_l1_to_l2", "earth_mars")
    s = truncated_setup(; case_id, objective = :quantile, chance_formulation = :slack)
    model, n = s.blocks.model, C.n_arcs(s.ref)
    @test model.meta.nvar == 63n + 28
    @test model.meta.ncon == 42n + 34
    @test s.blocks.objective_constant == 0
    rng = Xoshiro(9)
    x = Array(model.meta.x0) .* (1 .+ 1e-3 .* randn(rng, model.meta.nvar))
    chance_rows = 35n + 34 .+ (1:n)
    @test abs(NLPModels.obj(model, x) - sum(Array(NLPModels.cons(model, x))[chance_rows])) < 1e-13 * NLPModels.obj(model, x)
    g = Array(NLPModels.grad(model, x))
    @test maximum(abs, g - central_differences(v -> [NLPModels.obj(model, v)], x)[1, :]) < 1e-6 * max(1, maximum(abs, g))
    y = randn(rng, model.meta.ncon)
    H = dense_hessian(model, x, y)
    Hfd = central_differences(v -> lagrangian_gradient(model, v, y), x)
    @test maximum(abs, H - (Hfd + Hfd') / 2) / max(1, maximum(abs, H)) < 1e-5
end
