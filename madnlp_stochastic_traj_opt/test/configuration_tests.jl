# Configuration: the execution modes, the output layout and option validation.

@testset "configuration" begin
    runs = C.parse_cli(["--mode=gpu", "--case=all", "--tol=1e-7", "--quiet"])
    @test [o.case_id for o in runs] == collect(C.CASE_IDS)
    @test all(o -> o.execution_mode == :gpu && !o.verbose, runs)
    @test runs[1].condensed_relaxation == 1e-7 / 10          # default: tol / 10
    @test C.output_directory(runs[2]) == joinpath(C.PROJECT_ROOT, "output", "gpu", "nrho_l2_to_dro")
    @test C.output_directory(C.Options()) == joinpath(C.PROJECT_ROOT, "output", "cpu", "lyapunov_l1_to_l2")
    cpu = C.stochastic_solver_options(C.Options())
    @test cpu.kkt_system == C.MadNLP.SparseKKTSystem && cpu.bound_relax_factor == 0.0
    @test redirect_stdout(() -> C.parse_cli(["--help"]), devnull) === nothing
    @test_throws ArgumentError C.parse_cli(["--mode=tpu"])
    @test_throws ArgumentError C.parse_cli(["--unknown=1"])
    @test_throws ArgumentError C.validate_options(C.Options(condensed_relaxation = 1e-5))
    @test_throws ArgumentError C.validate_options(C.Options(max_consecutive_restoration = 6))
end
