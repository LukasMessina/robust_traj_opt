using Test, LinearAlgebra, Random
import NLPModels
include(joinpath(@__DIR__, "..", "src", "DoubleIntegratorStochastic.jl"))
using .DoubleIntegratorStochastic
const DI = DoubleIntegratorStochastic

@testset "ShARK and augmented UT" begin
    rng = Xoshiro(164)
    for h in (0.05, 0.25, 0.5)
        o = Options(dt=h, tf=20h, make_plots=false, run_monte_carlo=false)
        d = DI.system_matrices(o)
        Qc,Lc = DI.calibrate_continuous_diffusion(d.Ac,d.G*d.G',h)
        @test DI.accumulated_process_covariance(d.Ac,Qc,h) ≈ d.G*d.G' atol=1e-15
        @test DI.validate_shark(o,d.Ac,Qc,Lc) < 1e-13
        x,u,W,H = randn(rng,4),randn(rng,2),sqrt(h)*randn(rng,4),sqrt(h/12)*randn(rng,4)
        y = DI.shark_step(x,u,W,H,Lc,h)
        # Independent exact linear SDE formula, only used as a test oracle.
        expected = d.A*x+d.B*u+Lc*W+h*d.Ac*Lc*(W/2+H)
        @test y ≈ expected atol=1e-14
        # General two-stage ShARK tableau, independent of scalar specialization.
        f1 = d.Ac*(x+Lc*H)+d.Bc*u
        y2 = x+(5h/6)*f1+(5/6)*Lc*W+Lc*H
        expected_tableau = x+h*((2/5)*f1+(3/5)*(d.Ac*y2+d.Bc*u))+Lc*W
        @test y ≈ expected_tableau atol=1e-14
        for kappa in (0.0, 1.0, -1.0)
            opt = Options(dt=h,tf=20h,scaling_parameter=kappa,make_plots=false,run_monte_carlo=false)
            Z,w = DI.unscented_rule(opt)
            @test size(Z) == (12,25)
            @test sum(w) ≈ 1
            @test Z*Diagonal(w)*Z' ≈ I atol=1e-14
            raw = randn(rng,4,4)
            P = raw*raw'+0.2I
            L = Matrix(cholesky(Symmetric(P)).L)
            K = 0.2randn(rng,2,4)
            mu,Pnext,S = DI.propagation_arc(opt,Lc,x,L,u,K)
            F = d.A+d.B*K
            @test mu ≈ d.A*x+d.B*u atol=1e-13
            @test Pnext ≈ F*P*F'+d.G*d.G' atol=1e-12
            @test S ≈ K*P*K' atol=1e-12
            @test DI.unpack_lower(DI.pack_lower(L)) == L
        end
    end
end

@testset "Control chance kernel and exact derivatives" begin
    rng = Xoshiro(514)
    o = Options(make_plots=false, run_monte_carlo=false)
    _, Lc = DI.calibrate_continuous_diffusion(DI.system_matrices(o).Ac, 1e-4I(4), o.dt)
    L = Matrix(cholesky(Symmetric(o.x0_covariance)).L)
    K = randn(rng,2,4)
    u = randn(rng,2)
    local_x = [u; vec(K); DI.pack_lower(L)]
    _, _, S = DI.propagation_arc(o,Lc,o.x0_mean,L,u,K)
    psi = sqrt(DI.quantile(DI.Chisq(2),o.control_confidence))
    parameters = (sqrt(12.0),1/24,psi,o.control_norm_epsilon,o.spectral_eigenvalue_smoothing)
    expected = DI.control_norm(u...,o.control_norm_epsilon) +
        psi*DI.spectral_radius_sqrt(S[1,1],S[2,2],S[1,2],o.spectral_eigenvalue_smoothing)
    indices = reshape(collect(1:20),20,1)
    values, gradient, hessian = zeros(1), zeros(20), zeros(210)
    rows = [r for r in 1:20 for c in 1:r]
    cols = [c for r in 1:20 for c in 1:r]
    DI.launch_chance!(DI.chance_value_kernel!,values,local_x,indices,parameters)
    DI.launch_chance!(DI.chance_jacobian_kernel!,gradient,local_x,indices,parameters)
    DI.launch_chance!(DI.chance_hessian_kernel!,hessian,local_x,[1.0],indices,rows,cols,parameters)
    @test values[1] ≈ expected atol=1e-12
    fun(x) = DI.local_control_chance(x,parameters...)
    @test gradient ≈ DI.ForwardDiff.gradient(fun,local_x) atol=1e-12
    dense_hessian = DI.ForwardDiff.hessian(fun,local_x)
    @test hessian ≈ [dense_hessian[r,c] for (r,c) in zip(rows,cols)] atol=1e-11
end

@testset "Options and reproducible rollouts" begin
    @test_throws ArgumentError DI.validate_options(Options(tf=4.0))
    @test_throws ArgumentError DI.validate_options(Options(execution_mode=:unknown))
    @test_throws ArgumentError DI.validate_options(Options(spectral_eigenvalue_smoothing=1e-3))
    @test_throws ArgumentError DI.validate_options(Options(monte_carlo_samples=100))
    @test_throws ArgumentError DI.validate_options(Options(max_consecutive_restoration=3))
    @test DI.relaxation_schedule(Options(execution_mode=:gpu_condensed)) == [1e-9]
    @test DI.relaxation_schedule(Options(execution_mode=:gpu_condensed,condensed_continuation=true)) == [1e-4,1e-6,1e-9]
    @test DI.relaxation_schedule(Options(execution_mode=:gpu_condensed,condensed_continuation=false)) == [1e-9]
    @test DI.relaxation_schedule(Options(execution_mode=:gpu_condensed,condensed_relaxation=1e-5)) == [1e-5]
    @test_throws ArgumentError DI.validate_options(Options(initial_gain_max_abs=-1e-5))
    @test_throws ArgumentError DI.validate_options(Options(initial_gain_uniform=NaN))
    @test_throws ArgumentError DI.validate_options(Options(initial_gain_uniform=1e-5,initial_gain_max_abs=1e-5))
    @test Options(execution_mode=:gpu_condensed,initial_gain_uniform=1e-5).initial_gain_max_abs === nothing
    @test DI.relaxation_schedule(Options(execution_mode=:cpu)) == [0.0]
    o=Options(monte_carlo_samples=256,make_plots=false)
    a,b=DI.rollout_inputs(o),DI.rollout_inputs(o)
    @test a.initial == b.initial
    @test a.paths == b.paths
    @test all(isfinite,a.paths)
    @test size(a.paths) == (8,20,256)
end

@testset "Uniform gain seed and covariance repropagation" begin
    o = Options(make_plots=false,run_monte_carlo=false)
    d = DI.system_matrices(o)
    _,Lc = DI.calibrate_continuous_diffusion(d.Ac,d.G*d.G',o.dt)
    nominal = (;means=repeat(o.x0_mean,1,o.n_arcs+1),feedforward=zeros(2,o.n_arcs))
    seed = DI.build_seed(o,d.A,d.B,Lc,nominal)
    @test size(seed.gains) == (2,4,20)
    @test all(==(1e-5),seed.gains)
    for mode in (:cpu,:gpu_condensed,:gpu_hybrid)
        defaults = Options(execution_mode=mode,make_plots=false,run_monte_carlo=false)
        default_seed = DI.build_seed(defaults,d.A,d.B,Lc,nominal)
        @test all(==(1e-5),default_seed.gains)
        @test defaults.initial_gain_max_abs === nothing
        @test defaults.alpha_min_frac == 0.05
    end
    zero_seed = DI.build_seed(Options(warm_start_gains=false),d.A,d.B,Lc,nominal)
    @test all(iszero,zero_seed.gains)
    tvlqr_options = Options(initial_gain_uniform=nothing)
    tvlqr_seed = DI.build_seed(tvlqr_options,d.A,d.B,Lc,nominal)
    @test tvlqr_seed.gains == DI.tvlqr_gains(tvlqr_options,d.A,d.B)
    P = copy(o.x0_covariance)
    # Independent linear covariance formula is an oracle only; build_seed uses UT.
    F = d.A+d.B*fill(1e-5,2,4)
    for k in 1:o.n_arcs
        P = F*P*F'+d.G*d.G'
        L = DI.unpack_lower(seed.cholesky_factor[:,k+1])
        @test L*L' ≈ P atol=1e-12
    end
end

@testset "Command-line options and output location" begin
    options = DI.parse_cli_options(["--mode=gpu_condensed", "--no-plots", "--no-monte-carlo"])
    @test options.execution_mode == :gpu_condensed
    @test options.alpha_min_frac == 0.05
    @test options.initial_gain_uniform == 1e-5
    @test !options.make_plots && !options.run_monte_carlo
    @test options.output_dir == joinpath(DI.PROJECT_ROOT,"output","gpu_condensed")
    @test DI.parse_cli_options(["--alpha-min-frac=0"]).alpha_min_frac == 0.0
    tvlqr = DI.parse_cli_options(["--initial-gain-uniform=none", "--tvlqr-control-weight-scale=1"])
    @test tvlqr.initial_gain_uniform === nothing
    @test tvlqr.initial_gain_max_abs === nothing
    @test tvlqr.tvlqr_control_weight_scale == 1.0
    @test_throws ArgumentError DI.parse_cli_options(["--unknown"])
    @test_throws ArgumentError DI.parse_cli_options(["--initial-gain-max-abs=1e-5"])
    @test_throws ArgumentError DI.parse_cli_options(["--alpha-min-frac=-0.05"])
end

@testset "Consecutive restoration guard" begin
    guard=DI.RestorationGuard(limit=2)
    @test DI.record_restoration!(guard,4,:soft)
    @test DI.record_restoration!(guard,5,:robust)
    @test guard.max_consecutive == 2
    @test guard.restoration_iterations == 2
    # A regular callback without a completed step must not reset the run.
    @test guard((;cnt=(;k=6)),DI.MadNLP.UserCallbackRegular())
    @test !DI.allow_restoration!(guard,6,:soft)
    @test guard.limit_exceeded
    @test guard.restoration_iterations == 2
    # An intervening completed regular iteration does break the run.
    @test DI.record_restoration!(guard,7,:robust)
    @test guard.consecutive == 1
    @test !DI.allow_restoration!(DI.RestorationGuard(limit=0),0,:soft)
end

@testset "Native restoration stops before a third step" begin
    c=DI.ExaModels.ExaCore(Float64)
    DI.ExaModels.@add_var(c,x,1:1;start=0.0)
    DI.ExaModels.@add_obj(c,x[1]^2)
    DI.ExaModels.@add_con(c,x[1];lcon=1.0,ucon=Inf)
    model=DI.ExaModels.ExaModel(c)
    for limit in (0,1,2)
        settings=DI.solver_options(Options(print_level=0,max_consecutive_restoration=limit))
        solver=DI.MadNLP.MadNLPSolver(model;settings...)
        DI.MadNLP.initialize!(solver)
        # Prevent return to REGULAR so the native restoration loop exercises
        # the guard, independent of how quickly this small problem improves.
        empty!(solver.filter)
        push!(solver.filter,(0.0,-Inf))
        status=DI.MadNLP.robust!(solver)
        guard=solver.intermediate_callback
        @test status == DI.MadNLP.USER_REQUESTED_STOP
        @test guard.restoration_iterations == limit
        @test guard.max_consecutive == limit
        @test guard.limit_exceeded
    end
    # Soft callbacks occur after the accepted step. The second callback must
    # stop the loop when the native filter would demand a third step.
    solver=DI.MadNLP.MadNLPSolver(model;DI.solver_options(Options(print_level=0))...)
    DI.MadNLP.initialize!(solver)
    empty!(solver.filter); push!(solver.filter,(0.0,-Inf))
    guard=solver.intermediate_callback
    DI.record_restoration!(guard,0,:soft)
    solver.cnt.k=2
    @test !guard(solver,DI.MadNLP.UserCallbackRestore())
    @test guard.restoration_iterations == 2
    @test guard.limit_exceeded
end

@testset "UT moment kernels" begin
    rng = Xoshiro(784)
    o = Options(make_plots=false,run_monte_carlo=false)
    _,Lc = DI.calibrate_continuous_diffusion(DI.system_matrices(o).Ac,1e-4I(4),o.dt)
    raw = randn(rng,4,4)
    L = Matrix(cholesky(Symmetric(raw*raw'+0.1I)).L)
    K = 0.1randn(rng,2,4)
    x = [vec(K);DI.pack_lower(L)]
    indices = reshape(collect(1:18),18,1)
    Z,w = DI.unscented_rule(o)
    z,gw,gh = Z[1:4,:],Lc*(sqrt(o.dt)*Z[5:8,:]),Lc*(sqrt(o.dt/12)*Z[9:12,:])
    values,jac,hess = zeros(14),zeros(252),zeros(171)
    pr = [r for r in 1:18 for c in 1:r]
    pc = [c for r in 1:18 for c in 1:r]
    multipliers = randn(rng,14)
    DI.launch_moment!(DI.moment_value_kernel!,values,1,x,indices,z,gw,gh,w,o.dt,1)
    DI.launch_moment!(DI.moment_jacobian_kernel!,jac,18,x,indices,z,gw,gh,w,o.dt)
    DI.launch_moment!(DI.moment_hessian_kernel!,hess,171,x,multipliers,indices,pr,pc,z,gw,gh,w,o.dt,1)
    mu,P,_ = DI.propagation_arc(o,Lc,zeros(4),L,zeros(2),K)
    @test values ≈ -[mu;DI.pack_lower(P)] atol=1e-12
    fun(x) = collect(DI.local_ut_moments(x,z,gw,gh,w,o.dt))
    @test reshape(jac,14,18) ≈ DI.ForwardDiff.jacobian(fun,x) atol=1e-12
    H = DI.ForwardDiff.hessian(x -> dot(multipliers,fun(x)),x)
    @test hess ≈ [H[r,c] for (r,c) in zip(pr,pc)] atol=1e-11
end

# ExaModels AD is checked against directional finite differences. There are
# genuine nonlinear K*L products and nonlinear chance margins in this model.
function check_derivatives(model)
    rng=Xoshiro(34)
    x=Array(model.meta.x0)
    # Avoid degenerate seed covariances: finite differences near the small
    # eigenvalue smoothing scale can be inaccurate.
    # Feasibility is unnecessary for a derivative check.
    x .+= 0.01randn(rng,length(x))
    v=randn(rng,length(x)); v ./= norm(v)
    h=2e-6
    grad=NLPModels.grad(model,x)
    @test dot(grad,v) ≈ (NLPModels.obj(model,x+h*v)-NLPModels.obj(model,x-h*v))/(2h) rtol=1e-5 atol=1e-7
    J=NLPModels.jac(model,x)
    @test J*v ≈ (NLPModels.cons(model,x+h*v)-NLPModels.cons(model,x-h*v))/(2h) rtol=1e-5 atol=2e-7
    y=randn(rng,model.meta.ncon)
    H=Symmetric(NLPModels.hess(model,x,y),:L)
    gp=NLPModels.grad(model,x+h*v)+NLPModels.jac(model,x+h*v)'*y
    gm=NLPModels.grad(model,x-h*v)+NLPModels.jac(model,x-h*v)'*y
    hv_fd = (gp-gm)/(2h)
    println("Lagrangian Hessian directional relative error: ",norm(H*v-hv_fd)/max(1,norm(hv_fd)))
    @test H*v ≈ hv_fd rtol=2e-4 atol=2e-5
end

if "--model" in ARGS || "--solve" in ARGS
    @testset "ExaModels transcription and derivatives" begin
        o=Options(print_level=0,run_monte_carlo=false,make_plots=false)
        d=DI.system_matrices(o)
        _,Lc=DI.calibrate_continuous_diffusion(d.Ac,d.G*d.G',o.dt)
        nominal_blocks=DI.build_nominal_model(o,nothing)
        guard_settings=merge(DI.solver_options(o;nominal=true),
            (;intermediate_callback=DI.RestorationGuard(limit=0)))
        guarded_solver=DI.MadNLP.MadNLPSolver(nominal_blocks.model;guard_settings...)
        iteration_before=guarded_solver.cnt.k
        @test DI.MadNLP.restore!(guarded_solver) == DI.MadNLP.USER_REQUESTED_STOP
        @test guard_settings.intermediate_callback.blocked_phase == :soft
        @test guarded_solver.cnt.k == iteration_before
        @test DI.MadNLP.robust!(guarded_solver) == DI.MadNLP.USER_REQUESTED_STOP
        @test guard_settings.intermediate_callback.blocked_phase == :robust
        @test guarded_solver.cnt.k == iteration_before
        nominal_stats=DI.MadNLP.madnlp(nominal_blocks.model;DI.solver_options(o;nominal=true)...)
        nominal=(means=Array(DI.ExaModels.solution(nominal_stats,nominal_blocks.means)),
            feedforward=Array(DI.ExaModels.solution(nominal_stats,nominal_blocks.feedforward)))
        seed=DI.build_seed(o,d.A,d.B,Lc,nominal)
        blocks=DI.build_stochastic_model(o,Lc,seed,nothing)
        @test blocks.model.meta.nvar == 504
        @test blocks.model.meta.ncon == 456
        @test count(blocks.model.meta.lcon .== blocks.model.meta.ucon) == 308
        check_derivatives(blocks.model)
        if "--solve" in ARGS
            run=DI.solve_stochastic_model(blocks.model,o)
            stats=run.stats
            @test stats.status == DI.MadNLP.SOLVE_SUCCEEDED
            @test run.guard.blocked_phase == :none
            @test run.total_iterations <= o.max_iter
            @test DI.original_violation(blocks.model,stats.solution) <= 1e-8
            sol=DI.extract_solution(stats,blocks)
            diag=DI.diagnostics(o,Lc,sol)
            @test diag.objective ≈ stats.objective rtol=1e-8
            @test diag.mean_defect <= 1e-8
            @test diag.covariance_defect <= 1e-8
        end
    end
end
