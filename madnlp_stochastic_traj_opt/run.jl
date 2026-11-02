#!/usr/bin/env julia
# julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/run.jl [--mode=cpu|gpu] [--case=...] [--help]
# Exit status 1 when any run fails the reported convergence/feasibility requirements.
using CR3BPStochastic

if abspath(PROGRAM_FILE) == @__FILE__
    CR3BPStochastic.main() || exit(1)
end
