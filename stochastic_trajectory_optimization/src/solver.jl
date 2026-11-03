# MadNLP 0.10 reports soft restoration after an accepted step and robust
# restoration before a step. Account for that difference when counting runs.
# Dispatch is specialized only for solvers carrying this application's guard.
Base.@kwdef mutable struct RestorationGuard <: MadNLP.AbstractUserCallback
    limit::Int = typemax(Int)     # consecutive restoration iterations allowed (no limit; the report checks <= 5)
    regular_iterations::Int = 0
    restoration_iterations::Int = 0
    consecutive::Int = 0
    max_consecutive::Int = 0
    last_restoration_iteration::Int = -2
    iteration_offset::Int = 0
    blocked_phase::Symbol = :none
    limit_exceeded::Bool = false
    flush_stdout::Bool = false
end

next_restoration_count(g::RestorationGuard,k) =
    k == g.last_restoration_iteration+1 ? g.consecutive+1 : 1

function allow_restoration!(g::RestorationGuard,k,phase)
    if next_restoration_count(g,k) > g.limit
        g.blocked_phase = phase
        g.limit_exceeded = true
        return false
    end
    return true
end

function record_restoration!(g::RestorationGuard,k,phase)
    g.flush_stdout && flush(stdout)
    allow_restoration!(g,k,phase) || return false
    g.consecutive = next_restoration_count(g,k)
    g.max_consecutive = max(g.max_consecutive,g.consecutive)
    g.last_restoration_iteration = k
    g.restoration_iterations += 1
    return true
end

function (g::RestorationGuard)(solver,::MadNLP.UserCallbackRegular)
    g.flush_stdout && flush(stdout)
    g.regular_iterations = solver.cnt.k
    # A callback precedes the regular step, which could fail without advancing
    # k. Only an actual gap in iteration indices breaks a restoration run.
    return true
end

function (g::RestorationGuard)(solver,::MadNLP.UserCallbackRestore)
    k = g.iteration_offset + solver.cnt.k - 1
    record_restoration!(g,k,:soft) || return false
    if g.consecutive == g.limit
        # The native routine checks this exact condition immediately after its
        # callback. Permit its return to REGULAR, but never its next loop step.
        theta = MadNLP.get_theta(MadNLP.get_c(solver))
        varphi = MadNLP.get_varphi(MadNLP.get_obj_val(solver),
            MadNLP.get_x_lr(solver),MadNLP.get_xl_r(solver),
            MadNLP.get_xu_r(solver),MadNLP.get_x_ur(solver),MadNLP.get_mu(solver))
        if !MadNLP.is_filter_acceptable(MadNLP.get_filter(solver),theta,varphi)
            g.blocked_phase = :soft
            g.limit_exceeded = true
            return false
        end
    end
    return true
end

function (g::RestorationGuard)(solver,::MadNLP.UserCallbackRobust)
    return record_restoration!(g,g.iteration_offset+solver.cnt.k,:robust)
end

const GuardedMadNLPSolver = MadNLP.MadNLPSolver{T,VT,VI,KKT,Model,CB,Iterator,IC,KV,RestorationGuard} where
    {T,VT,VI,KKT,Model,CB,Iterator,IC,KV}

# Keep native restoration compilation lazy: most runs never need these large
# routines, especially their GPU specializations. This is a function barrier,
# not a different restoration algorithm.
native_soft_restoration!(solver) = invoke(MadNLP.restore!,Tuple{MadNLP.AbstractMadNLPSolver},solver)
native_robust_restoration!(solver) = invoke(MadNLP.robust!,Tuple{MadNLP.AbstractMadNLPSolver},solver)

function MadNLP.restore!(solver::GuardedMadNLPSolver)
    g = solver.intermediate_callback
    allow_restoration!(g,g.iteration_offset+solver.cnt.k,:soft) || return MadNLP.USER_REQUESTED_STOP
    return Base.invokelatest(native_soft_restoration!,solver)::MadNLP.Status
end

function MadNLP.robust!(solver::GuardedMadNLPSolver)
    g = solver.intermediate_callback
    allow_restoration!(g,g.iteration_offset+solver.cnt.k,:robust) || return MadNLP.USER_REQUESTED_STOP
    return Base.invokelatest(native_robust_restoration!,solver)::MadNLP.Status
end

# MadNLP configuration. Both execution modes solve the same KKT system with the same settings: the exact
# Lagrangian Hessian, the monotone barrier rule, SparseKKTSystem + MUMPS and exact equalities, on the CPU; the gpu
# mode only evaluates the model on a CUDA device (`HostModel`). The deterministic reoptimisations run on the CPU.

const ALPHA_MIN_FRAC = 0.05                  # MadNLP alpha_min_frac
const BARRIER_MU_INIT = 0.1                  # stochastic solve, monotone barrier rule
const BARRIER_MU_MIN = 1e-10
# Least-squares initial multipliers are kept up to this infinity norm and otherwise all set to zero
# (MadNLP's default 1e3). With zero multipliers the first step sees only the objective's curvature.
const CONSTR_MULT_INIT_MAX = 1e4
const NOMINAL_TOL = 1e-8                     # deterministic reoptimisations
const NOMINAL_MAX_ITER = 3000
const NOMINAL_BARRIER_MU_INIT = 0.1
const PSYCHE_NOMINAL_BARRIER_MU_INIT = 1e-6  # its already-nearly-feasible seed needs a smaller initial barrier

"Array backend of the stochastic model: `nothing` (CPU arrays) or the CUDA backend."
function execution_backend(o::Options)
    o.execution_mode == :cpu && return nothing
    @eval import CUDA   # loaded only for GPU runs
    return Base.invokelatest(cuda_backend)
end

function cuda_backend()
    CUDA.functional() || error("CUDA is unavailable; use execution_mode = :cpu")
    CUDA.allowscalar(false)
    return CUDA.CUDABackend()
end

const LINEAR_SYSTEM = (; linear_solver = MadNLP.MumpsSolver, kkt_system = MadNLP.SparseKKTSystem,
    equality_treatment = MadNLP.EnforceEquality, bound_relax_factor = 0.0)

function solver_options(o::Options; tol, acceptable_tol, mu_init, mu_min, max_iter, output_file)
    return (; tol, acceptable_tol, acceptable_iter = 15, max_iter,
        hessian_approximation = MadNLP.ExactHessian,
        barrier = MadNLP.MonotoneUpdate{Float64}(; mu_init, mu_min),
        alpha_min_frac = ALPHA_MIN_FRAC,
        intermediate_callback = RestorationGuard(),
        print_level = o.verbose ? MadNLP.INFO : MadNLP.ERROR,
        output_file, file_print_level = MadNLP.INFO, LINEAR_SYSTEM...)
end

"Single-phase reoptimisation: tighter tolerance."
nominal_solver_options(o::Options; output_file::String = "") = solver_options(o;
    tol = NOMINAL_TOL, acceptable_tol = NOMINAL_TOL, mu_init = NOMINAL_BARRIER_MU_INIT,
    mu_min = min(1e-4, NOMINAL_TOL) / 11, max_iter = NOMINAL_MAX_ITER, output_file)

stochastic_solver_options(o::Options; output_file::String = "") = merge(solver_options(o;
    tol = o.tol, acceptable_tol = o.acceptable_tol, mu_init = BARRIER_MU_INIT, mu_min = BARRIER_MU_MIN,
    max_iter = o.max_iter, output_file), (; constr_mult_init_max = CONSTR_MULT_INIT_MAX))

"Psyche reoptimisation: the effective MadNLP settings of the verified runs."
function psyche_nominal_solver_options(o::Options; output_file::String = "")
    settings = solver_options(o; tol = NOMINAL_TOL, acceptable_tol = NOMINAL_TOL, mu_init = PSYCHE_NOMINAL_BARRIER_MU_INIT,
        mu_min = min(1e-4, NOMINAL_TOL) / 11, max_iter = NOMINAL_MAX_ITER, output_file)
    # MadNLP uses bound_push/bound_fac for both variables and slacks; bound_fac is its default 1e-2.
    return merge(settings, (bound_push = 1e-8, bound_fac = 1e-2))
end

"Build and run a MadNLP solver; returns `(stats, solver)`."
function run_madnlp(model, settings)
    # ExaModels encodes expression graphs in its types; these function barriers keep
    # them out of the driver.
    solver = Base.invokelatest(MadNLP.MadNLPSolver, model; settings...)
    stats = Base.invokelatest(MadNLP.solve!, solver)
    return stats, solver
end

"Final KKT measures exactly as MadNLP's own summary reports them."
function kkt_measures(solver)
    obj_scale = MadNLP.get_cb(solver).obj_scale[]
    inf_du, inf_pr, inf_compl = MadNLP.get_inf_du(solver), MadNLP.get_inf_pr(solver), MadNLP.get_inf_compl(solver)
    constraint_scaled = norm(MadNLP.get_c(solver), Inf)
    return Dict{String,Any}(
        "dual_infeasibility_scaled" => inf_du,
        "dual_infeasibility_unscaled" => inf_du / obj_scale,
        "constraint_violation_scaled" => constraint_scaled,
        "constraint_violation_unscaled" => inf_pr,
        "complementarity_scaled" => inf_compl * obj_scale,
        "complementarity_unscaled" => inf_compl,
        "overall_nlp_error_scaled" => max(inf_du * obj_scale, constraint_scaled, inf_compl),
        "overall_nlp_error_unscaled" => max(inf_du, inf_pr, inf_compl),
        "objective_scale" => obj_scale)
end

"Largest violation of the original (unrelaxed) constraints and variable bounds."
function original_violation(model, x)
    values = Array(NLPModels.cons(model, x))
    xv = Array(x)
    return max(maximum(max.(Array(model.meta.lcon) .- values, values .- Array(model.meta.ucon), 0.0)),
        maximum(max.(Array(model.meta.lvar) .- xv, xv .- Array(model.meta.uvar), 0.0)))
end

# GPU execution (execution_mode = :gpu): the model is built and evaluated on the device, MadNLP and
# its linear solver (MUMPS, the CPU's exact sparse KKT system) run on the host. `HostModel` is the host view
# MadNLP sees: every evaluation copies the point (and the multipliers) to the device, evaluates the device
# model there and copies the values back; the sparsity structures are copied once. MadNLP passes views of
# its own arrays, so every transfer goes through a contiguous host buffer (a device copy into a view would
# fall back to scalar indexing).

struct HostModel{M<:NLPModels.AbstractNLPModel,VD} <: NLPModels.AbstractNLPModel{Float64,Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64,Vector{Float64}}
    counters::NLPModels.Counters
    device::M
    x::VD            # device copies of the point, the multipliers and the outputs
    y::VD
    c::VD
    g::VD
    jac::VD
    hess::VD
    v::VD
    jv::VD
    host_x::Vector{Float64}      # the same on the host
    host_y::Vector{Float64}
    host_c::Vector{Float64}
    host_g::Vector{Float64}
    host_jac::Vector{Float64}
    host_hess::Vector{Float64}
    host_v::Vector{Float64}
    host_jv::Vector{Float64}
    jac_rows::Vector{Int}
    jac_cols::Vector{Int}
    hess_rows::Vector{Int}
    hess_cols::Vector{Int}
end

function HostModel(device::NLPModels.AbstractNLPModel)
    m = device.meta
    meta = NLPModels.NLPModelMeta(m.nvar; x0 = Array(m.x0), lvar = Array(m.lvar), uvar = Array(m.uvar), ncon = m.ncon,
        y0 = Array(m.y0), lcon = Array(m.lcon), ucon = Array(m.ucon), nnzj = m.nnzj, nnzh = m.nnzh, lin = Int[],
        minimize = m.minimize, name = m.name)
    sizes = (m.nvar, m.ncon, m.ncon, m.nvar, m.nnzj, m.nnzh, m.nvar, m.ncon)
    buffers = [fill!(similar(m.x0, n), 0.0) for n in sizes]
    rows, cols = similar(m.x0, Int, m.nnzj), similar(m.x0, Int, m.nnzj)
    NLPModels.jac_structure!(device, rows, cols)
    hrows, hcols = similar(m.x0, Int, m.nnzh), similar(m.x0, Int, m.nnzh)
    NLPModels.hess_structure!(device, hrows, hcols)
    return HostModel(meta, NLPModels.Counters(), device, buffers..., [zeros(n) for n in sizes]...,
        Array(rows), Array(cols), Array(hrows), Array(hcols))
end

"Copy host `v` (possibly a view) to the device buffer `d` through the host buffer `h`."
to_device!(d, h, v::AbstractVector) = copyto!(d, copyto!(h, v))
"Copy the device buffer `d` to host `v` (possibly a view) through the host buffer `h`."
to_host!(v::AbstractVector, h, d) = copyto!(v, copyto!(h, d))

NLPModels.obj(m::HostModel, x::AbstractVector) = NLPModels.obj(m.device, to_device!(m.x, m.host_x, x))

function NLPModels.grad!(m::HostModel, x::AbstractVector, g::AbstractVector)
    NLPModels.grad!(m.device, to_device!(m.x, m.host_x, x), m.g)
    return to_host!(g, m.host_g, m.g)
end

function NLPModels.cons_nln!(m::HostModel, x::AbstractVector, c::AbstractVector)
    NLPModels.cons!(m.device, to_device!(m.x, m.host_x, x), m.c)
    return to_host!(c, m.host_c, m.c)
end

function NLPModels.jac_structure!(m::HostModel, rows::AbstractVector, cols::AbstractVector)
    copyto!(rows, m.jac_rows)
    copyto!(cols, m.jac_cols)
    return rows, cols
end

function NLPModels.jac_coord!(m::HostModel, x::AbstractVector, vals::AbstractVector)
    NLPModels.jac_coord!(m.device, to_device!(m.x, m.host_x, x), m.jac)
    return to_host!(vals, m.host_jac, m.jac)
end

function NLPModels.jprod_nln!(m::HostModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    NLPModels.jprod!(m.device, to_device!(m.x, m.host_x, x), to_device!(m.v, m.host_v, v), m.jv)
    return to_host!(Jv, m.host_jv, m.jv)
end

function NLPModels.jtprod_nln!(m::HostModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
    NLPModels.jtprod!(m.device, to_device!(m.x, m.host_x, x), to_device!(m.jv, m.host_jv, v), m.v)
    return to_host!(Jtv, m.host_v, m.v)
end

function NLPModels.hess_structure!(m::HostModel, rows::AbstractVector, cols::AbstractVector)
    copyto!(rows, m.hess_rows)
    copyto!(cols, m.hess_cols)
    return rows, cols
end

function NLPModels.hess_coord!(m::HostModel, x::AbstractVector, y::AbstractVector, vals::AbstractVector;
        obj_weight = one(eltype(x)))
    NLPModels.hess_coord!(m.device, to_device!(m.x, m.host_x, x), to_device!(m.y, m.host_y, y), m.hess; obj_weight)
    return to_host!(vals, m.host_hess, m.hess)
end

NLPModels.hess_coord!(m::HostModel, x::AbstractVector, vals::AbstractVector; obj_weight = one(eltype(x))) =
    NLPModels.hess_coord!(m, x, zeros(m.meta.ncon), vals; obj_weight)

"The model MadNLP solves: the CPU model itself, or the host view of the device model in the gpu mode."
solver_model(o::Options, model) = o.execution_mode == :gpu ? HostModel(model) : model
