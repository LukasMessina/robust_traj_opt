#!/usr/bin/env julia
# julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl [--mode=cpu|gpu] [--case=...] [--help]
# Exit status 1 when a run does not end with SOLVE_SUCCEEDED or breaks the restoration requirement.
include(joinpath(@__DIR__, "src", "CR3BPStochastic.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    CR3BPStochastic.main() || exit(1)
end
