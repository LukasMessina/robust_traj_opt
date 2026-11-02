# MadNLP configuration. Both execution modes use the exact Lagrangian Hessian and the
# monotone barrier rule; they differ only in the linear algebra:
#   cpu  SparseKKTSystem + MUMPS, exact equalities
#   gpu  SparseCondensedKKTSystem + cuDSS LDL, equalities relaxed by `condensed_relaxation`
# The energy-optimal seed is always solved on the CPU.

"Array backend of the stochastic model: `nothing` (CPU arrays) or the CUDA backend."
function execution_backend(o::Options)
    o.execution_mode == :cpu && return nothing
    @eval import CUDA, CUDSS, MadNLPGPU   # loaded only for GPU runs
    return Base.invokelatest(cuda_backend)
end

function cuda_backend()
    CUDA.functional() || error("CUDA is unavailable; use execution_mode = :cpu")
    CUDA.allowscalar(false)
    return CUDA.CUDABackend()
end

const CPU_LINEAR_SYSTEM = (; linear_solver = MadNLP.MumpsSolver, kkt_system = MadNLP.SparseKKTSystem,
    equality_treatment = MadNLP.EnforceEquality, bound_relax_factor = 0.0)

gpu_linear_system(o::Options) = (; linear_solver = MadNLPGPU.CUDSSSolver, cudss_algorithm = MadNLP.LDL,
    kkt_system = MadNLP.SparseCondensedKKTSystem, equality_treatment = MadNLP.RelaxEquality,
    fixed_variable_treatment = MadNLP.MakeParameter, bound_relax_factor = o.condensed_relaxation)

stochastic_linear_system(o::Options) = o.execution_mode == :gpu ? gpu_linear_system(o) : CPU_LINEAR_SYSTEM

"MadNLP barrier update of `rule` (one of BARRIER_RULES)."
function barrier_update(rule::Symbol, mu_init, mu_min)
    rule == :quality && return MadNLP.QualityFunctionUpdate{Float64}(; mu_init, mu_min)
    rule == :loqo && return MadNLP.LOQOUpdate{Float64}(; mu_init, mu_min)
    return MadNLP.MonotoneUpdate{Float64}(; mu_init, mu_min)
end

function solver_options(o::Options; tol, acceptable_tol, mu_init, mu_min, max_iter, output_file, linear_system,
        barrier_rule = :monotone)
    return (; tol, acceptable_tol, acceptable_iter = 15, max_iter,
        hessian_approximation = MadNLP.ExactHessian,
        barrier = barrier_update(barrier_rule, mu_init, mu_min),
        alpha_min_frac = o.alpha_min_frac,
        intermediate_callback = RestorationGuard(limit = something(o.max_consecutive_restoration, typemax(Int))),
        print_level = o.verbose ? MadNLP.INFO : MadNLP.ERROR,
        output_file, file_print_level = MadNLP.INFO, linear_system...)
end

"Energy-optimal seed solve: CPU, exact equalities, tighter tolerance."
nominal_solver_options(o::Options; output_file::String = "") = solver_options(o;
    tol = o.nominal_tol, acceptable_tol = o.nominal_tol, mu_init = o.nominal_barrier_mu_init,
    mu_min = min(1e-4, o.nominal_tol) / 11, max_iter = o.nominal_max_iter, output_file,
    linear_system = CPU_LINEAR_SYSTEM)

stochastic_solver_options(o::Options; output_file::String = "") = merge(solver_options(o;
    tol = o.tol, acceptable_tol = o.acceptable_tol, mu_init = o.barrier_mu_init, mu_min = o.barrier_mu_min,
    max_iter = o.max_iter, output_file, linear_system = stochastic_linear_system(o), barrier_rule = o.barrier_rule),
    (; constr_mult_init_max = o.constr_mult_init_max, bound_push = o.bound_push))

"Psyche energy seed: preserve the effective MadNLP settings of the verified runs."
function psyche_nominal_solver_options(o::Options; output_file::String = "")
    settings=solver_options(o;tol=o.nominal_tol,acceptable_tol=o.nominal_tol,
        mu_init=o.psyche_nominal_barrier_mu_init,mu_min=min(1e-4,o.nominal_tol)/11,
        max_iter=o.nominal_max_iter,output_file,linear_system=CPU_LINEAR_SYSTEM)
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
