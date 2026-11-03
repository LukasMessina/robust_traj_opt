# Startup checks only: no trajectory model, CUDA context, or optimisation run.
@testset "Psyche fuel reoptimisation settings" begin
    core = C.ExaCore(Float64)
    C.@add_var(core, x, 1:2; start=[0.,0.])
    C.@add_obj(core, x[i]^2 for i in 1:2)
    model = C.ExaModel(core)
    for mode in (:cpu, :gpu)
        o = C.Options(case_id="psyche", execution_mode=mode)
        settings = C.psyche_nominal_solver_options(o)
        accepted = mktemp() do _, io
            redirect_stdout(io) do
                C.MadNLP.load_options(model; settings...).interior_point
            end
        end
        @test accepted.bound_push == 1e-8
        @test accepted.bound_fac == 1e-2
        @test accepted.max_iter == C.NOMINAL_MAX_ITER
        @test accepted.kkt_system == C.MadNLP.SparseKKTSystem   # the reoptimisation always runs on the CPU
    end
end

@testset "Output root is resolved after package loading" begin
    o=C.Options(case_id="psyche",execution_mode=:cpu)
    withenv("MADNLP_STOCHASTIC_OUTPUT_ROOT"=>joinpath(tempdir(),"madnlp_runtime_root")) do
        @test C.output_directory(o)==joinpath(tempdir(),"madnlp_runtime_root","cpu","psyche")
    end
    withenv("MADNLP_STOCHASTIC_OUTPUT_ROOT"=>nothing) do
        @test C.output_directory(o)==joinpath(C.PROJECT_ROOT,"output","cpu","psyche")
    end
end
