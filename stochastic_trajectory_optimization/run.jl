#!/usr/bin/env julia
# julia --project=environment --threads=auto stochastic_trajectory_optimization/run.jl [--mode=cpu|gpu] [--case=...] [--help]
# Exit status 1 when any run fails the reported convergence/feasibility requirements.
using CR3BPStochastic

if abspath(PROGRAM_FILE) == @__FILE__
    CR3BPStochastic.main() || exit(1)
end
