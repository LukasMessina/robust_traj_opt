"""
CR3BP stochastic low-thrust trajectory optimization with ExaModels and MadNLP.

Julia port of `cr3bp_stochastic_traj_opt_jax.py`: the same normalized
multiple-shooting NLP (node means, feedforward, gains, seed-scaled covariance
Cholesky factors and the terminal margin factor), augmented unscented transform
with navigation error, XMDS2 RK9 acceleration diffusion and nested Gates
execution error, thrust chance and terminal covariance constraints, and the same
energy-optimal + TVLQR + propagated-covariance initial guess. Exact Hessians only;
runs on the CPU (MUMPS) or on a CUDA GPU (condensed KKT system, cuDSS).
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
include("problem.jl")
include("dynamics.jl")
include("unscented.jl")
include("reference.jl")
include("initialization.jl")
include("solver/restoration_guard.jl")
include("solver/madnlp.jl")
include("oracles/dual_numbers.jl")
include("nominal.jl")
include("oracles/ut_moments.jl")
include("oracles/chance.jl")
include("model.jl")
include("diagnostics.jl")
include("solve.jl")
include("output.jl")
include("cli.jl")

end # module
