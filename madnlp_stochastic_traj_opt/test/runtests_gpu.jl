using Test, TOML
include(joinpath(@__DIR__, "..", "src", "DoubleIntegratorStochastic.jl"))
using .DoubleIntegratorStochastic
const DI = DoubleIntegratorStochastic
include("gpu_kernel_tests.jl")
include("hybrid_kkt_tests.jl")
backend=DI.execution_backend(Options(execution_mode=:gpu_condensed))
Base.invokelatest(check_gpu_kernels,backend)
"--kernels-only" in ARGS || Base.invokelatest(check_gpu_transcription,backend)
if !("--kernels-only" in ARGS)
    hybrid_backend=DI.execution_backend(Options(execution_mode=:gpu_hybrid))
    Base.invokelatest(check_hybrid_kkt,hybrid_backend)
end

# Requires a working CUDA device. Both modes run in one Julia process so their
# shared device kernels are compiled once. CPU tests are in runtests.jl.
modes = any(arg -> arg in ("--kernels-only","--derivatives-only"),ARGS) ? () : isempty(ARGS) ? (:gpu_condensed,:gpu_hybrid) : Tuple(Symbol.(ARGS))
objectives = Float64[]
@testset "GPU solver modes" begin
for mode in modes
    @testset "$mode original problem" begin
        result = solve_problem(Options(execution_mode=mode,run_monte_carlo=false,
            make_plots=false,print_level=2))
        DI.save_results(result)
        @test result.report["converged"]
        @test result.report["blocked_restoration_request"] == "none"
        @test result.report["max_consecutive_restoration"] <= 2
        @test !result.report["restoration_limit_exceeded"]
        @test result.report["nvar"] == 504
        @test result.report["ncon"] == 456
        @test result.report["max_constraint_violation"] <= 1e-8
        @test result.solution.independent_violation <= 1e-8
        @test result.solution.objective ≈ result.stats.objective rtol=1e-9
        push!(objectives,result.solution.objective)
    end
end
end
if length(objectives) > 1
    # Equality bands perturb the objective by their multipliers times the band.
    # Feasibility is checked independently at 1e-8 above.
    @test maximum(objectives)-minimum(objectives) < 1e-6
end
cpu_report = joinpath(DI.PROJECT_ROOT,"output","cpu","diagnostics.toml")
if isfile(cpu_report)
    saved = TOML.parsefile(cpu_report)
    defaults = Options()
    if saved["spectral_eigenvalue_smoothing"] == defaults.spectral_eigenvalue_smoothing &&
        saved["problem"]["terminal_margin_floor"] == defaults.terminal_margin_floor
        @test all(isapprox(value,saved["objective"];atol=1e-6,rtol=0) for value in objectives)
    elseif !isempty(objectives)
        println("Saved CPU report uses different smoothing or terminal floor; skipping its objective comparison.")
    end
end
