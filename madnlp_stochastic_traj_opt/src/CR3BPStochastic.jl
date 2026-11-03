"""
CR3BP, Earth--Mars and multiphase Psyche stochastic low-thrust trajectory optimization
with ExaModels and MadNLP.

The normalized multiple-shooting NLP of `stochastic_traj_opt_jax.py` (node means, throttle and
direction, gains, seed-scaled covariance Cholesky factors), augmented unscented transform with
navigation error, XMDS2 RK9 acceleration diffusion and nested Gates execution error, thrust chance
constraints and the terminal covariance, from the fuel-optimal reference + TVLQR +
propagated-covariance initial guess; two problems (`OBJECTIVES`, options.jl). Exact Hessians only;
MadNLP with MUMPS on the CPU, the model evaluated on the CPU or on a CUDA GPU.
Mission code is precompilable; CUDA is still loaded only for GPU runs.
"""
module CR3BPStochastic

using LinearAlgebra
using Printf
using TOML
using StaticArrays
using Distributions: Chisq, quantile
using ExaModels
using KernelAbstractions
import ForwardDiff
import NLPModels
import MadNLP
import NPZ

export Options, solve_problem, save_results, main

const PROJECT_ROOT = dirname(@__DIR__)
const REPOSITORY_ROOT = dirname(PROJECT_ROOT)

const NX = 7   # augmented state: position, velocity, mass
const NU = 3   # control
const NP = 6   # position/velocity, observed by the feedback
const NW = 3   # acceleration Wiener processes
const NL = 28  # packed 7x7 lower factor
const NM = 21  # packed 6x6 lower factor

include("generated/tableaux.jl")
include("generated/cases.jl")
include("options.jl")
include("packing.jl")
include("dynamics.jl")
include("problem.jl")
include("unscented.jl")
include("reference.jl")
include("initialization.jl")
include("solver/restoration_guard.jl")
include("solver/madnlp.jl")
include("solver/host_model.jl")
include("oracles/dual_numbers.jl")
include("nominal.jl")
include("oracles/ut_moments.jl")
include("oracles/ut_factor.jl")
include("oracles/chance.jl")
include("oracles/chance_slack.jl")
include("model.jl")
include("diagnostics.jl")
include("solve.jl")
include("output.jl")
include("psyche/data.jl")
include("psyche/geometry.jl")
include("psyche/nominal.jl")
include("psyche/initialization.jl")
include("psyche/model.jl")
include("psyche/solve.jl")
include("validation.jl")
include("cli.jl")
include("precompile_workload.jl")

end # module
