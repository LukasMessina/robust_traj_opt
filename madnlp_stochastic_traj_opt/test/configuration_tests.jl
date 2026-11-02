# Configuration: the execution modes, the output layout and option validation.

@testset "configuration" begin
    runs = C.parse_cli(["--mode=gpu", "--case=all", "--tol=1e-7", "--quiet"])
    @test [o.case_id for o in runs] == collect(C.CASE_IDS)
    @test all(o -> o.execution_mode == :gpu && !o.verbose, runs)
    @test runs[1].condensed_relaxation == 1e-7 / 100         # default: tol / 100
    @test C.output_directory(runs[2]) == joinpath(C.PROJECT_ROOT, "output", "gpu", "nrho_l2_to_dro_terminal_cost_factor_throttle")
    @test C.output_directory(C.Options()) == joinpath(C.PROJECT_ROOT, "output", "cpu", "lyapunov_l1_to_l2_terminal_cost_factor_throttle")
    cpu = C.stochastic_solver_options(C.Options())
    @test cpu.kkt_system == C.MadNLP.SparseKKTSystem && cpu.bound_relax_factor == 0.0
    @test redirect_stdout(() -> C.parse_cli(["--help"]), devnull) === nothing
    @test_throws ArgumentError C.parse_cli(["--mode=tpu"])
    @test_throws ArgumentError C.parse_cli(["--unknown=1"])
    @test_throws ArgumentError C.validate_options(C.Options(condensed_relaxation = 1e-5))
    @test_throws ArgumentError C.validate_options(C.Options(max_consecutive_restoration = 6))
    psyche=only(C.parse_cli(["--case=psyche"]))
    @test psyche.initial_position_std_km==2500.
    @test psyche.psyche_terminal_position_std_km==1500. && psyche.psyche_terminal_velocity_std_km_s==0.002
    @test psyche.control_covariance_weight==C.case_default("psyche",:control_covariance_weight)
    @test only(C.parse_cli(["--case=psyche","--max-iter=10000","--nominal-max-iter=10000"])).nominal_max_iter==10000
    # the defaults: the step 3 configuration (throttle, fuel reference x2, terminal cost, factor defects, delta 5e-4)
    lyapunov = C.Options()
    @test lyapunov.control_parameterization == :throttle && lyapunov.reference_objective == :fuel &&
        lyapunov.reference_mesh_refinement == 2 && lyapunov.terminal_covariance == :cost && lyapunov.covariance_defect == :factor &&
        lyapunov.model_mass_flow_smoothing == 5e-4 && lyapunov.gain_bound == Inf && lyapunov.constr_mult_init_max == 1e4
    @test psyche.psyche_nominal_objective == :fuel && psyche.reference_mesh_refinement === nothing &&
        psyche.state_covariance_weight == 0.0102 && psyche.control_covariance_weight == 54.3
    @test C.Options(truncated_uniform_mesh_arcs = 3).reference_mesh_refinement === nothing
    @test only(C.parse_cli(["--case=psyche","--psyche-terminal-position-std-km=150"])).psyche_terminal_position_std_km==150.
    @test only(C.parse_cli(["--case=psyche","--initial-position-std-km=3000"])).initial_position_std_km==3000.
    @test_throws ArgumentError C.parse_cli(["--case=psyche","--psyche-terminal-velocity-std-km-s=0"])
    @test_throws ArgumentError C.parse_cli(["--case=psyche","--chance-formulation=slack","--control-parameterization=cartesian",
        "--terminal-covariance=constraint"])
    @test only(C.parse_cli(["--case=psyche","--chance-formulation=slack"])).chance_formulation == :slack   # throttle: the slack oracle
    @test_throws ArgumentError C.parse_cli(["--case=psyche","--psyche-substeps=2"])
    # Post-solve Monte Carlo validation on the device of the solve.
    @test psyche.validate_after_solve
    @test !only(C.parse_cli(["--no-validation"])).validate_after_solve
    gpu=only(C.parse_cli(["--case=psyche","--mode=gpu","--validation-python=/opt/py"]))
    @test C.validation_command(gpu).exec==["/opt/py",joinpath(C.PROJECT_ROOT,"tools","validate_psyche.py"),
        C.output_directory(gpu),"--mode=gpu"]
    cpu=only(C.parse_cli(["--case=earth_mars","--validation-python=/opt/py"]))
    @test C.validation_command(cpu).exec==["/opt/py",joinpath(C.PROJECT_ROOT,"tools","validate.py"),"cpu","--case=earth_mars",
        "--directory=$(C.output_directory(cpu))"]
    hybrid=only(C.parse_cli(["--case=earth_mars","--mode=hybrid","--validation-python=/opt/py"]))
    @test C.validation_command(hybrid).exec[3]=="gpu"   # hybrid: JAX on the GPU
    @test C.validation_input(only(C.parse_cli(["--case=earth_mars","--feasibility","--control-parameterization=cartesian",
        "--terminal-covariance=constraint","--reference-objective=energy"]))) === nothing
end
