# Included by runtests_gpu.jl, which defines DI and imports Test.
function check_gpu_kernels(backend)
    @testset "CPU/GPU exact derivative kernels" begin
        N=2
        cv(a)=DI.ExaModels.convert_array(a,backend)
        # Deterministic nonsymmetric input exercises every gain/factor entry.
        x=sin.(collect(1:40))
        indices=reshape(collect(1:40),20,N)
        p=(sqrt(12.0),1/24,3.4,1e-6,1e-5)
        rows=[r for r in 1:20 for c in 1:r]
        cols=[c for r in 1:20 for c in 1:r]
        for (kernel,size,args) in (
            (DI.chance_value_kernel!,N,(x,indices,p)),
            (DI.chance_jacobian_kernel!,20N,(x,indices,p)),
            (DI.chance_hessian_kernel!,210N,(x,[0.3,0.7],indices,rows,cols,p)))
            cpu=zeros(size);gpu=cv(cpu)
            gpuargs=map(a -> a isa AbstractArray ? cv(a) : a,args)
            DI.launch_chance!(kernel,cpu,args...)
            DI.launch_chance!(kernel,gpu,gpuargs...)
            @test Array(gpu) ≈ cpu rtol=1e-11 atol=1e-11
        end
        o=DI.Options(make_plots=false,run_monte_carlo=false)
        d=DI.system_matrices(o)
        _,Lc=DI.calibrate_continuous_diffusion(d.Ac,d.G*d.G',o.dt)
        Z,w=DI.unscented_rule(o)
        z,gw,gh=Z[1:4,:],Lc*(sqrt(o.dt)*Z[5:8,:]),Lc*(sqrt(o.dt/12)*Z[9:12,:])
        x=sin.(collect(1:36))
        indices=reshape(collect(1:36),18,N)
        rows=[r for r in 1:18 for c in 1:r]
        cols=[c for r in 1:18 for c in 1:r]
        for (kernel,size,ndrange,args) in (
            (DI.moment_value_kernel!,14N,N,(x,indices,z,gw,gh,w,o.dt,N)),
            (DI.moment_jacobian_kernel!,252N,18N,(x,indices,z,gw,gh,w,o.dt)),
            (DI.moment_hessian_kernel!,171N,171N,(x,cos.(collect(1:14N)),indices,rows,cols,z,gw,gh,w,o.dt,N)))
            cpu=zeros(size);gpu=cv(cpu)
            gpuargs=map(a -> a isa AbstractArray ? cv(a) : a,args)
            DI.launch_moment!(kernel,cpu,ndrange,args...)
            DI.launch_moment!(kernel,gpu,ndrange,gpuargs...)
            @test Array(gpu) ≈ cpu rtol=1e-11 atol=1e-11
        end
    end
end

function check_gpu_transcription(backend)
    @testset "CPU/GPU complete transcription" begin
        o=DI.Options(print_level=0,make_plots=false,run_monte_carlo=false)
        d=DI.system_matrices(o)
        _,Lc=DI.calibrate_continuous_diffusion(d.Ac,d.G*d.G',o.dt)
        nb=DI.build_nominal_model(o,nothing)
        ns=DI.run_madnlp(nb.model,o,DI.solver_options(o;nominal=true))
        nominal=(means=Array(DI.ExaModels.solution(ns,nb.means)),feedforward=Array(DI.ExaModels.solution(ns,nb.feedforward)))
        seed=DI.build_seed(o,d.A,d.B,Lc,nominal)
        cpu=DI.build_stochastic_model(o,Lc,seed,nothing).model
        gpu=DI.build_stochastic_model(o,Lc,seed,backend).model
        x=cpu.meta.x0+0.01sin.(collect(1:cpu.meta.nvar))
        y=cos.(collect(1:cpu.meta.ncon))
        gx,gy=DI.ExaModels.convert_array(x,backend),DI.ExaModels.convert_array(y,backend)
        @test DI.NLPModels.obj(gpu,gx) ≈ DI.NLPModels.obj(cpu,x) rtol=1e-12
        @test Array(DI.NLPModels.grad(gpu,gx)) ≈ DI.NLPModels.grad(cpu,x) rtol=1e-11 atol=1e-11
        @test Array(DI.NLPModels.cons(gpu,gx)) ≈ DI.NLPModels.cons(cpu,x) rtol=1e-11 atol=1e-11
        function coords(model,x,y)
            jr=similar(x,Int,model.meta.nnzj);jc=similar(jr)
            hr=similar(x,Int,model.meta.nnzh);hc=similar(hr)
            j=similar(x,model.meta.nnzj);h=similar(x,model.meta.nnzh)
            DI.NLPModels.jac_structure!(model,jr,jc)
            DI.NLPModels.hess_structure!(model,hr,hc)
            DI.NLPModels.jac_coord!(model,x,j)
            DI.NLPModels.hess_coord!(model,x,y,h)
            return map(Array,(jr,jc,hr,hc,j,h))
        end
        reference,device=coords(cpu,x,y),coords(gpu,gx,gy)
        for i in 1:4
            @test device[i] == reference[i]
        end
        @test device[5] ≈ reference[5] rtol=1e-10 atol=1e-10
        @test device[6] ≈ reference[6] rtol=1e-10 atol=1e-10
    end
end
