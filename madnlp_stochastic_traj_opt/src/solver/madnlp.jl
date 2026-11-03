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
