# Whole-model checks on a truncated problem (3 arcs, CPU): constraint values
# against the literal UT arc map, the objective against the literal nested-UT
# costs, the Jacobian against central differences of the constraints, and the
# Lagrangian Hessian (native algebra + both oracles) against central differences
# of the exact Lagrangian gradient.

@testset "stochastic model derivatives" begin
    s = truncated_setup()
    model, n = s.blocks.model, C.n_arcs(s.ref)
    @test model.meta.nvar == 56n + 28
    @test model.meta.ncon == 36n + 34
    rng = Xoshiro(3)
    x = Array(model.meta.x0) .* (1 .+ 1e-3 .* randn(rng, model.meta.nvar))

    sol = C.extract_solution(x, s.blocks, s.scaling, s.seed, n)
    values = Array(NLPModels.cons(model, x))
    offset = C.NX + C.NP   # boundary rows precede the defects
    for k in 1:n
        mu, P, _ = C.propagation_arc(s.p, sol.means[:, k], sol.covariances[:, :, k], sol.feedforward[:, k],
            sol.gains[:, :, k], s.ref.steps[k])
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
