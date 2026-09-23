using Test, LinearAlgebra

function check_hybrid_kkt(backend)
    @testset "Regularized hybrid linear systems" begin
        c=DI.ExaModels.ExaCore(Float64;backend)
        DI.ExaModels.@add_var(c,x,1:4;start=[0.1,0.2,0.3,0.4],lvar=-2.0,uvar=2.0)
        DI.ExaModels.@add_obj(c,0.5i*x[i]^2 for i in 1:4)
        DI.ExaModels.@add_con(c,x[1]+x[2]-1.0)
        DI.ExaModels.@add_con(c,x[3]+2x[4]-0.5)
        DI.ExaModels.@add_con(c,x[1]+x[3];lcon=-1.0,ucon=2.0)
        DI.ExaModels.@add_con(c,x[2]-x[4];lcon=-Inf,ucon=1.0)
        model=DI.ExaModels.ExaModel(c)
        o=DI.Options(execution_mode=:gpu_hybrid,print_level=0)
        settings=merge(DI.solver_options(o),(;kkt_system=DI.RegularizedHybridKKTSystem))
        solver=DI.MadNLP.MadNLPSolver(model;settings...)
        solver.kkt.gamma[]=1e3
        DI.MadNLP.initialize!(solver)
        kkt=solver.kkt
        DI.MadNLP.set_aug_diagonal!(kkt,solver)
        DI.MadNLP.regularize_diagonal!(kkt,1.0,0.0)
        p,d,w=DI.MadNLP.get_p(solver),DI.MadNLP.get_d(solver),DI.MadNLP.get__w4(solver)
        rhs=sin.(collect(1:length(DI.MadNLP.full(p))))
        for diagonal in ([0.,0.,0.,0.],[-1e-8,-2e-8,0.,0.],
            [0.,0.,-0.1,-0.2],[-0.01,-0.02,-0.03,-0.04])
            copyto!(kkt.du_diag,diagonal)
            DI.MadNLP.build_kkt!(kkt)
            DI.MadNLP.factorize!(kkt.linear_solver)
            copyto!(DI.MadNLP.full(p),rhs)
            copyto!(DI.MadNLP.full(d),rhs)
            DI.MadNLP.solve_kkt!(kkt,d)
            fill!(DI.MadNLP.full(w),0.0)
            # The original HybridKKT multiplication independently represents
            # the full, uncondensed system, including its dual diagonal.
            mul!(w,kkt.inner,d,1.0,0.0)
            residual=norm(Array(DI.MadNLP.full(w))-rhs,Inf)/norm(rhs,Inf)
            @test residual < 1e-8
            @test minimum(Array(kkt.equality_scale)) >= 0.5-1e-14
        end
    end
end
