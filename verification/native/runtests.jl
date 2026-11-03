# CPU checks:  julia --project=environment --threads=auto verification/native/runtests.jl
# Add --gpu for the CUDA checks (not while another GPU job is running: 8 GB device).
include(joinpath(@__DIR__, "..", "..", "stochastic_trajectory_optimization", "src", "CR3BPStochastic.jl"))
using .CR3BPStochastic
include(joinpath(@__DIR__, "setup.jl"))

include(joinpath(@__DIR__, "configuration_tests.jl"))
include(joinpath(@__DIR__, "psyche_startup_tests.jl"))
include(joinpath(@__DIR__, "dynamics_tests.jl"))
include(joinpath(@__DIR__, "model_tests.jl"))
isfile(C.PSYCHE_INPUT_FILE) && include(joinpath(@__DIR__, "psyche_tests.jl"))
"--gpu" in ARGS && include(joinpath(@__DIR__, "gpu_tests.jl"))
