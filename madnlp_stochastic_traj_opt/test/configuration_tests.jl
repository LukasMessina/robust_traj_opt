# Configuration: the two problems, the execution modes, the output layout and option validation.

@testset "configuration" begin
    runs = C.parse_cli(["--mode=gpu", "--case=all", "--quiet"])
    @test [o.case_id for o in runs] == collect(C.CASE_IDS)
    @test all(o -> o.execution_mode == :gpu && !o.verbose && o.objective == :standard && o.max_iter == 5000, runs)
    @test C.output_directory(runs[2]) == joinpath(C.PROJECT_ROOT, "output", "gpu", "nrho_l2_to_dro")
    @test C.output_directory(C.Options()) == joinpath(C.PROJECT_ROOT, "output", "cpu", "lyapunov_l1_to_l2")
    settings = C.stochastic_solver_options(C.Options())
    @test settings.kkt_system == C.MadNLP.SparseKKTSystem && settings.linear_solver == C.MadNLP.MumpsSolver &&
        settings.bound_relax_factor == 0.0 && settings.constr_mult_init_max == 1e4
    @test redirect_stdout(() -> C.parse_cli(["--help"]), devnull) === nothing
    @test_throws ArgumentError C.parse_cli(["--mode=tpu"])
    @test_throws ArgumentError C.parse_cli(["--mode=hybrid"])
    @test_throws ArgumentError C.parse_cli(["--objective=quantile"])
    @test_throws ArgumentError C.parse_cli(["--unknown=1"])
    @test_throws ArgumentError C.validate_options(C.Options(case_id = "psyche", truncated_arcs = 3))

    # step 4: chance-sum cost, terminal bound, Bryson seed weights, 10000 iterations, its own output directory
    step4 = only(C.parse_cli(["--case=earth_mars", "--objective=chance_sum"]))
    @test C.chance_sum(step4) && step4.max_iter == 10000 && C.seed_weights(step4) == (1.0, 1.0)
    @test C.output_directory(step4) == joinpath(C.PROJECT_ROOT, "output", "cpu", "earth_mars_chance_sum")
    step3 = only(C.parse_cli(["--case=earth_mars", "--max-iter=10000"]))
    @test !C.chance_sum(step3) && step3.max_iter == 10000 &&
        C.seed_weights(step3) == (step3.state_covariance_weight, step3.control_covariance_weight)

    psyche = only(C.parse_cli(["--case=psyche"]))
    @test psyche.initial_position_std_km == 2500.
    @test psyche.psyche_terminal_position_std_km == 1500. && psyche.psyche_terminal_velocity_std_km_s == 0.002
    @test psyche.state_covariance_weight == 0.0102 && psyche.control_covariance_weight == 54.3
    @test C.seed_weights(C.Options(case_id = "psyche", objective = :chance_sum)) == (0.0102, 54.3)
    @test_throws ArgumentError C.validate_options(C.Options(case_id = "psyche", psyche_terminal_velocity_std_km_s = 0.0))

    # Post-solve Monte Carlo validation on the device of the solve.
    @test psyche.validate_after_solve
    @test !only(C.parse_cli(["--no-validation"])).validate_after_solve
    gpu = only(C.parse_cli(["--case=psyche", "--mode=gpu", "--validation-python=/opt/py"]))
    @test C.validation_command(gpu).exec == ["/opt/py", joinpath(C.PROJECT_ROOT, "tools", "validate_psyche.py"),
        C.output_directory(gpu), "--mode=gpu"]
    cpu = only(C.parse_cli(["--case=earth_mars", "--validation-python=/opt/py"]))
    @test C.validation_command(cpu).exec == ["/opt/py", joinpath(C.PROJECT_ROOT, "tools", "validate.py"), "cpu", "--case=earth_mars",
        "--directory=$(C.output_directory(cpu))"]
end
