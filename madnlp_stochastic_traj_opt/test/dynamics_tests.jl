# Exact-derivative building blocks. First derivatives against ForwardDiff;
# forward-over-reverse Hessians against central differences of the exact gradient
# (nested ForwardDiff through the unrolled RK9 takes hours to compile).

@testset "dynamics derivatives" begin
    rng = Xoshiro(1)
    case = C.case_data("lyapunov_l1_to_l2")
    mu, ve = case.mu, case.ve
    x = SVector{7}(0.86, 0.12, 0.01, 0.09, 0.04, -0.002, 0.98)
    u = SVector{3}(0.03, -0.1, 0.02)
    w = SVector{7}(randn(rng, 7))
    f(v) = Vector(C.cr3bp_vector_field(SVector{7}(v[1:7]), SVector{3}(v[8:10]), v[11], mu, ve))
    J = ForwardDiff.jacobian(f, [x; u; norm(u)])
    sb, cb, mb = C.cr3bp_vjp(x, u, norm(u), w, mu, ve)
    @test maximum(abs, J' * w - [sb; cb; mb]) < 1e-12 * max(1, maximum(abs, J' * w))
    for TB in (C.Dopri8, C.XMDS2RK9)
        h = 0.0575
        gdw = C.diffusion_increment(2.3e-3, SVector(0.3, -0.1, 0.2) * sqrt(h))
        z0 = [x; u]
        step(v) = C.rk_step(TB, SVector{7}(v[1:7]), SVector{3}(v[8:10]), norm(SVector{3}(v[8:10])), gdw, h, mu, ve)
        grad = ForwardDiff.gradient(v -> dot(w, step(v)), z0)
        y1, _ = C.rk_step_stages(TB, x, u, norm(u), gdw, h, mu, ve)
        @test y1 ≈ step(z0) atol = 1e-15
        function adjoint_gradient(v)
            xs, us = SVector{7}(v[1:7]), SVector{3}(v[8:10])
            _, stages = C.rk_step_stages(TB, xs, us, norm(us), gdw, h, mu, ve)
            xb, ub, mbar = C.rk_adjoint(TB, stages, us, norm(us), w, h, mu, ve)
            return [xb; ub + mbar * us / norm(us)]
        end
        @test maximum(abs, adjoint_gradient(z0) - grad) < 1e-12 * max(1, maximum(abs, grad))
        xd, ud = C.seed(C.TagPoint, x, 0, Val(10)), C.seed(C.TagPoint, u, 7, Val(10))
        _, stages = C.rk_step_stages(TB, xd, ud, norm(ud), gdw, h, mu, ve)
        xbd, ubd, mbd = C.rk_adjoint(TB, stages, ud, norm(ud), w, h, mu, ve)
        zb = vcat(xbd, ubd + mbd * ud / norm(ud))
        H = [C.partial(zb[r], q) for r in 1:10, q in 1:10]
        @test maximum(abs, H - central_differences(adjoint_gradient, z0)) < 1e-6 * max(1, maximum(abs, H))
        @test maximum(abs, H - H') < 1e-10 * max(1, maximum(abs, H))
    end
end
