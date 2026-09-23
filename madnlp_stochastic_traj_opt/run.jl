#!/usr/bin/env julia
# Run with: julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/run.jl
include(joinpath(@__DIR__, "src", "DoubleIntegratorStochastic.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    DoubleIntegratorStochastic.main()
end
