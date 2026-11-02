# Compare a fresh process loading source with the compiled package. Run each
# mode in separate Julia processes, with the same --threads setting.
using TOML
mode=isempty(ARGS) ? "package" : ARGS[1]
start=time_ns()
if mode=="source"
    include(joinpath(@__DIR__,"..","src","CR3BPStochastic.jl"))
else
    @eval using CR3BPStochastic
end
loaded=(time_ns()-start)/1e9
const C=CR3BPStochastic
start=time_ns()
C.precompile_single_phase_workload("lyapunov_l1_to_l2")
C.precompile_single_phase_workload("earth_mars")
first=(time_ns()-start)/1e9
start=time_ns()
C.precompile_single_phase_workload("lyapunov_l1_to_l2")
C.precompile_single_phase_workload("earth_mars")
warmed=(time_ns()-start)/1e9
report=Dict("mode"=>mode,"load_seconds"=>loaded,"first_workload_seconds"=>first,
    "load_and_first_workload_seconds"=>loaded+first,
    "warm_workload_seconds"=>warmed,"julia"=>string(VERSION),"threads"=>Threads.nthreads(),"optimisation_run"=>false)
TOML.print(stdout,report)
if length(ARGS)>1
    open(ARGS[2],"w") do io;TOML.print(io,report);end
end
