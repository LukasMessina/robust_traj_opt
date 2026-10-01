# CUDA kernels against the CPU model (validated by model_tests.jl) on the same
# truncated problem: objective, gradient, constraints, Jacobian and Lagrangian
# Hessian values, entry by entry in the shared sparsity order.

# Loads CUDA into the module; a separate top-level statement so the new methods are visible.
const GPU_BACKEND = C.execution_backend(C.Options(execution_mode = :gpu))

@testset "GPU kernels match the CPU model" begin
    cpu = truncated_setup()
    gpu = truncated_setup(; backend = GPU_BACKEND)
    mc, mg = cpu.blocks.model, gpu.blocks.model
    rng = Xoshiro(5)
    x = Array(mc.meta.x0) .* (1 .+ 1e-3 .* randn(rng, mc.meta.nvar))
    y = randn(rng, mc.meta.ncon)
    xg, yg = copyto!(similar(mg.meta.x0), x), copyto!(similar(mg.meta.x0, mc.meta.ncon), y)
    rel(a, b) = maximum(abs, Array(a) - Array(b)) / max(1, maximum(abs, Array(b)))
    @test rel(NLPModels.cons(mg, xg), NLPModels.cons(mc, x)) < 1e-12
    @test abs(NLPModels.obj(mg, xg) - NLPModels.obj(mc, x)) < 1e-12 * abs(NLPModels.obj(mc, x))
    @test rel(NLPModels.grad(mg, xg), NLPModels.grad(mc, x)) < 1e-12
    @test sparsity(mg, true) == sparsity(mc, true)
    @test rel(NLPModels.jac_coord(mg, xg), NLPModels.jac_coord(mc, x)) < 1e-11
    @test sparsity(mg, false) == sparsity(mc, false)
    @test rel(NLPModels.hess_coord(mg, xg, yg), NLPModels.hess_coord(mc, x, y)) < 1e-10
end
