#!/usr/bin/env julia
# Fixed-candidate profiling only: never runs trajectory optimisation or writes
# production results. CUDA timings include explicit synchronisation.
const LOAD_BEGIN=time_ns()
using CR3BPStochastic
const PACKAGE_LOAD_SECONDS=(time_ns()-LOAD_BEGIN)/1e9
const CUDA_BEGIN=time_ns()
using CUDA,LinearAlgebra,Statistics,TOML,SHA
const CUDA_LOAD_SECONDS=(time_ns()-CUDA_BEGIN)/1e9
const C=CR3BPStochastic

arg(name,default)=let matches=filter(a->startswith(a,name*"="),ARGS)
    isempty(matches) ? default : split(last(matches),'=';limit=2)[2]
end
const CASE=arg("--case","psyche")
const REPEATS=parse(Int,arg("--repeats","5"))
const OUT=abspath(arg("--output",joinpath(C.PROJECT_ROOT,"output","profile",CASE)))
mkpath(OUT)
const REPORT=Dict{String,Any}("case"=>CASE,"optimisation_run"=>false,
    "package_load_seconds"=>PACKAGE_LOAD_SECONDS,"cuda_import_seconds"=>CUDA_LOAD_SECONDS,
    "julia_version"=>string(VERSION),"threads"=>Threads.nthreads(),"substeps"=>1,
    "blas_threads"=>BLAS.get_num_threads(),"passed"=>false)
save()=open(joinpath(OUT,"profile.toml"),"w") do io;TOML.print(io,REPORT);end
say(s)=(println(s);flush(stdout);save())

function timed(f;gpu=false,repeats=REPEATS)
    gpu && CUDA.synchronize()
    t=time_ns();f();gpu && CUDA.synchronize();cold=(time_ns()-t)/1e9
    wall=Float64[];device=Float64[]
    for _ in 1:repeats
        gpu && CUDA.synchronize()
        t=time_ns();f();gpu && CUDA.synchronize();push!(wall,(time_ns()-t)/1e9)
        if gpu
            push!(device,CUDA.@elapsed f())
            CUDA.synchronize()
        end
    end
    r=Dict("first_call_seconds"=>cold,"wall_seconds"=>wall,"median_wall_seconds"=>median(wall))
    gpu && merge!(r,Dict("device_event_seconds"=>device,"median_device_event_seconds"=>median(device)))
    return r
end

function build_pair(o,backend)
    if CASE=="psyche"
        d=C.load_psyche_data(o.psyche_input_file);u=C.psyche_uncertainty(d,o);params=C.psyche_phase_params(d,o,u)
        REPORT["commands"]=count(>(0),d.slots);REPORT["intervals"]=d.n
        path=abspath(arg("--seed",joinpath(C.PROJECT_ROOT,"output","gpu","psyche","psyche_seed.npz")))
        s=C.NPZ.npzread(path)
        size(s["gains"],3)==d.n || error("seed mesh does not match mission inputs")
        REPORT["seed_path"]=path;REPORT["seed_sha256"]=bytes2hex(sha256(read(path)))
        REPORT["mission_input_sha256"]=d.sha256
        # A frozen candidate is used to isolate evaluation cost. It is not
        # asserted to be a feasible solution of the selected integration grid.
        ref=(means=s["means"],feedforward=s["feedforward"],days=s["days"],steps=C.arc_steps(d,s["days"]))
        seed=C.InitialGuess(ref.means,ref.feedforward,s["gains"],s["gain_scale"],s["margin"],s["covariances"],zeros(3,3,d.n))
        std=s["covariance_std"]
        sc=C.CovarianceScaling(std.^2,std,1 ./std,s["factors"],s["initial_factor"])
        t=time_ns();cpu=C.psyche_stochastic_model(d,o,u,params,ref,seed,sc,nothing)
        REPORT["cpu_model_build_seconds"]=(time_ns()-t)/1e9
        t=time_ns();gpu=C.psyche_stochastic_model(d,o,u,params,ref,seed,sc,backend);CUDA.synchronize()
        REPORT["gpu_model_build_seconds"]=(time_ns()-t)/1e9
    else
        case=C.case_data(CASE);norm=C.build_normalization(case,o);p=C.ut_params(case,o,norm)
        t=time_ns();ref=C.load_reference(case,o);seed=C.build_initial_guess(case,o,norm,p,ref);sc=C.covariance_scaling(seed,norm)
        REPORT["fixed_reference_seed_seconds"]=(time_ns()-t)/1e9
        REPORT["commands"]=length(ref.steps);REPORT["intervals"]=length(ref.steps)
        REPORT["sigma_points_per_command"]=p.n_points
        t=time_ns();cpu=C.build_stochastic_model(case,o,ref,seed,sc,p,nothing)
        REPORT["cpu_model_build_seconds"]=(time_ns()-t)/1e9
        t=time_ns();gpu=C.build_stochastic_model(case,o,ref,seed,sc,p,backend);CUDA.synchronize()
        REPORT["gpu_model_build_seconds"]=(time_ns()-t)/1e9
    end
    return cpu,gpu
end

function callbacks(model)
    x=copy(model.meta.x0);y=similar(x,model.meta.ncon);copyto!(y,sin.(1:model.meta.ncon))
    v=similar(x,model.meta.ncon);j=similar(x,model.meta.nnzj);h=similar(x,model.meta.nnzh)
    return (;x,y,v,j,h,
        objective=()->C.NLPModels.obj(model,x),
        constraints=()->C.NLPModels.cons!(model,x,v),
        jacobian=()->C.NLPModels.jac_coord!(model,x,j),
        hessian=()->C.NLPModels.hess_coord!(model,x,y,h))
end
relative(a,b)=maximum(abs.(a.-b))/max(1.,maximum(abs.(a)))

function activity_profile(f,stem)
    Base.invokelatest(f);CUDA.synchronize()
    result=nothing
    # The macro's own wrapper and CUPTI processing have first-use compilation.
    # Discard the first capture; otherwise that work appears as device idle time.
    for _ in 1:2
        result=CUDA.@profile external=false trace=true f()
    end
    open(joinpath(OUT,stem*".txt"),"w") do io
        show(IOContext(io,:limit=>false,:displaysize=>(100000,240)),MIME"text/plain"(),result)
    end
    # CUDA 6 returns the activity tables rather than printing inside the macro.
    # Retain raw event timing and resource information for independent inspection.
    for kind in (:host,:device)
        data=getproperty(result,kind)
        open(joinpath(OUT,stem*"_"*string(kind)*".tsv"),"w") do io
            println(io,join(string.(keys(data)),'\t'))
            for i in eachindex(data.start)
                println(io,join((replace(string(v[i]),'\t'=>' ','\n'=>' ') for v in values(data)),'\t'))
            end
        end
    end
    dev=result.device;host=result.host
    isempty(dev.start) && error("CUDA activity trace contains no device events")
    intervals=sort(collect(zip(dev.start,dev.stop)))
    lo,hi=first(intervals);busy=0.
    for (a,b) in Iterators.drop(intervals,1)
        if a>hi
            busy+=hi-lo;lo=a;hi=b
        else
            hi=max(hi,b)
        end
    end
    busy+=hi-lo
    span=maximum(dev.stop)-minimum(dev.start)
    copies=[i for i in eachindex(dev.name) if startswith(dev.name[i],"[copy ")]
    transfer=sum((dev.stop[i]-dev.start[i] for i in copies);init=0.)
    transferred=sum((dev.size[i] for i in copies if !ismissing(dev.size[i]));init=0)
    synchronization=sum((host.stop[i]-host.start[i] for i in eachindex(host.name) if occursin("synchron",lowercase(host.name[i])));init=0.)
    return Dict("trace"=>stem*".txt","device_events"=>length(dev.start),
        "device_span_seconds"=>span,"device_busy_union_seconds"=>busy,
        "device_gap_seconds"=>max(0.,span-busy),"device_transfer_sum_seconds"=>transfer,"transferred_bytes"=>transferred,
        "profiler_warmup_captures"=>1,
        "host_synchronization_sum_seconds"=>synchronization)
end

function run_profile()
    o=C.Options(case_id=CASE,execution_mode=:gpu,verbose=false)
    t=time_ns();backend=C.execution_backend(o)
    REPORT["backend_initialization_seconds"]=(time_ns()-t)/1e9
    REPORT["device"]=CUDA.name(CUDA.device())
    save()
    t=time_ns();cpu,gpu=Base.invokelatest(build_pair,o,backend)
    REPORT["total_model_setup_seconds"]=(time_ns()-t)/1e9;save()
    t=time_ns()
    Base.invokelatest(evaluate,cpu,gpu,o)
    REPORT["total_evaluation_validation_profiling_seconds"]=(time_ns()-t)/1e9;save()
end

function evaluate(cpu,gpu,o)
    a,b=callbacks(cpu.model),callbacks(gpu.model)
    REPORT["nvar"]=cpu.model.meta.nvar;REPORT["ncon"]=cpu.model.meta.ncon
    for name in (:objective,:constraints,:jacobian,:hessian)
        say("Timing $CASE $name (CPU, then GPU)")
        REPORT["cpu_"*string(name)]=timed(getproperty(a,name))
        save()
        REPORT["gpu_"*string(name)]=timed(getproperty(b,name);gpu=true)
        save()
    end
    parity=Dict("objective"=>abs(a.objective()-b.objective()),"constraints"=>relative(a.v,Array(b.v)),
        "jacobian"=>relative(a.j,Array(b.j)),"hessian"=>relative(a.h,Array(b.h)))
    all(v<1e-7 for v in values(parity)) || error("CPU/GPU parity failed: $parity")
    REPORT["cpu_gpu_relative_errors"]=parity;save()
    # Also check a nearby, generally infeasible point with nonzero multipliers.
    # Restore the fixed candidate before any assembly or KKT measurements.
    original=copy(a.x)
    a.x .*= 1 .+ 1e-6 .* sin.(eachindex(a.x))
    copyto!(b.x,a.x)
    for cb in (a,b)
        cb.constraints();cb.jacobian();cb.hessian()
    end
    perturbed=Dict("objective"=>abs(a.objective()-b.objective()),"constraints"=>relative(a.v,Array(b.v)),
        "jacobian"=>relative(a.j,Array(b.j)),"hessian"=>relative(a.h,Array(b.h)))
    all(v<1e-7 for v in values(perturbed)) || error("Perturbed CPU/GPU parity failed: $perturbed")
    REPORT["perturbed_cpu_gpu_relative_errors"]=perturbed
    copyto!(a.x,original);copyto!(b.x,original);save()
    say("Profiling warmed callbacks with CUDA activity tracing")
    try
        REPORT["cuda_activity_profile"]=activity_profile(()->begin
            b.constraints();b.jacobian();b.hessian();CUDA.synchronize()
        end,"cuda_trace")
    catch e
        REPORT["cuda_activity_profile_error"]=sprint(showerror,e)
    end
    # Profile KKT construction/factorisation at the initial candidate. No Newton
    # step or line search is taken, and no optimisation settings are changed.
    say("Timing initial KKT construction and factorisation")
    for (mode,model) in ((:cpu,cpu.model),(:gpu,gpu.model))
      key=string(mode)*"_"
      try
        options=C.Options(case_id=CASE,execution_mode=mode,verbose=false)
        settings=C.stochastic_solver_options(options)
        t=time_ns();solver=C.MadNLP.MadNLPSolver(model;settings...);CUDA.synchronize()
        REPORT[key*"solver_construction_seconds"]=(time_ns()-t)/1e9
        t=time_ns();C.MadNLP.initialize!(solver);CUDA.synchronize()
        REPORT[key*"solver_initialization_seconds"]=(time_ns()-t)/1e9
        kkt=C.MadNLP.get_kkt(solver)
        C.MadNLP.set_aug_diagonal!(kkt,solver)
        REPORT[key*"kkt_assembly"]=timed(()->C.MadNLP.build_kkt!(kkt);gpu=mode==:gpu)
        REPORT[key*"kkt_factorization"]=timed(()->C.MadNLP.factorize!(kkt.linear_solver);gpu=mode==:gpu)
        if mode==:gpu
            REPORT["cuda_kkt_profile"]=activity_profile(()->begin
                C.MadNLP.build_kkt!(kkt);C.MadNLP.factorize!(kkt.linear_solver);CUDA.synchronize()
            end,"cuda_kkt_trace")
        end
      catch e
        REPORT[key*"kkt_profile_error"]=sprint(showerror,e)
      end
      save()
    end
    REPORT["passed"]=true;save();println("Saved ",joinpath(OUT,"profile.toml"))
end
try
    run_profile()
catch e
    REPORT["error"]=sprint(showerror,e,catch_backtrace());save();rethrow()
end
