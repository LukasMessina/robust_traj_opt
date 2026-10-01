# CPU checks:  julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/test/runtests.jl
# Add --gpu for the CUDA checks (not while another GPU job is running: 8 GB device).
include(joinpath(@__DIR__, "..", "src", "CR3BPStochastic.jl"))
using .CR3BPStochastic
include(joinpath(@__DIR__, "setup.jl"))

include(joinpath(@__DIR__, "configuration_tests.jl"))
include(joinpath(@__DIR__, "dynamics_tests.jl"))
include(joinpath(@__DIR__, "model_tests.jl"))
"--gpu" in ARGS && include(joinpath(@__DIR__, "gpu_tests.jl"))
