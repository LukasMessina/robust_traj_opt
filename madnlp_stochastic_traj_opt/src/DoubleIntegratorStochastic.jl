"""UT-ShARK stochastic trajectory optimization with ExaModels and MadNLP."""
module DoubleIntegratorStochastic

using LinearAlgebra
using Random
using Statistics
using Printf
using TOML
using Distributions: Normal, Chisq, quantile
using ExaModels
using KernelAbstractions
import NLPModels
import MadNLP
import NPZ
import QuasiMonteCarlo

export Options, solve_problem, save_results, main

const PROJECT_ROOT = dirname(@__DIR__)

const NX = 4
const NU = 2
const NAUG = 12
const NSIGMA = 2NAUG + 1
const LOWER = [(r, c) for r in 1:NX for c in 1:r]
const DIAGONAL = [1, 3, 6, 10]
packed_index(r, c) = r * (r - 1) ÷ 2 + c

include("options.jl")
include("dynamics.jl")
include("restoration_guard.jl")
include("control_chance_oracle.jl")
include("ut_moment_oracle.jl")
include("solver.jl")
include("initialization.jl")
include("model.jl")
include("diagnostics.jl")
include("solve.jl")
include("postprocess.jl")
include("cli.jl")

end # module
