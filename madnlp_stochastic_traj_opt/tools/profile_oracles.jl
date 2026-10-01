# Timing of the NLP evaluations in one execution mode: the three UT oracle callbacks and
# the model's constraint, Jacobian and Lagrangian Hessian evaluations. Each is run once to
# compile, then REPS times with a device synchronization after each call. Model-level calls
# alternate between two points so that no evaluation cache can hide their cost.
#   julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/tools/profile_oracles.jl [cpu|gpu] [case] [reps]
include(joinpath(@__DIR__, "..", "src", "CR3BPStochastic.jl"))
using .CR3BPStochastic
using Printf, Statistics
import NLPModels
import KernelAbstractions
const C = CR3BPStochastic

const OPTIONS = C.validate_options(C.Options(; execution_mode = Symbol(get(ARGS, 1, "cpu")),
    case_id = get(ARGS, 2, "lyapunov_l1_to_l2")))
const REPS = parse(Int, get(ARGS, 3, "10"))
const BACKEND = C.execution_backend(OPTIONS)

function profile(o, backend)
    case = C.case_data(o.case_id)
    normalization = C.build_normalization(case)
    p = C.ut_params(case, o, normalization)
    ref = C.load_reference(case, o)
    seed = C.build_initial_guess(case, o, normalization, p, ref)
    scaling = C.covariance_scaling(seed, normalization)
    blocks = C.build_stochastic_model(case, o, ref, seed, scaling, p, backend)
    model, ut, n = blocks.model, blocks.ut, C.n_arcs(ref)
    sync() = KernelAbstractions.synchronize(KernelAbstractions.get_backend(model.meta.x0))
    x = copy(model.meta.x0)
    x2 = x .* (1 + 1e-9)
    y = fill!(similar(x, model.meta.ncon), 0.1)
    xl = x[1:model.meta.nvar-C.NM]            # the UT oracle's local vector (all but the margin)
    yl = fill!(similar(x, 35n), 0.1)
    out = similar(x, 35n)
    ut_jac = similar(x, length(ut.structure.jac_rows))
    ut_hess = similar(x, length(ut.structure.hess_rows))
    jac = similar(x, model.meta.nnzj)
    hess = similar(x, model.meta.nnzh)
    flip = Ref(false)
    point() = (flip[] = !flip[]; flip[] ? x2 : x)
    stages = [
        "UT values" => () -> C.ut_values!(out, xl, ut, p, n),
        "UT Jacobian" => () -> C.ut_jacobian!(ut_jac, xl, ut, p, n),
        "UT Hessian" => () -> C.ut_hessian!(ut_hess, xl, yl, ut, p, n),
        "model constraints" => () -> NLPModels.cons!(model, point(), y),
        "model Jacobian" => () -> NLPModels.jac_coord!(model, point(), jac),
        "model Hessian" => () -> NLPModels.hess_coord!(model, point(), y, hess),
    ]
    @printf("\n%s, %d arcs, %s, %d repetitions (times in ms)\n", o.case_id, n,
        backend === nothing ? "cpu ($(Threads.nthreads()) threads)" : "gpu", REPS)
    @printf("  %-20s %10s %10s %10s\n", "evaluation", "first", "min", "median")
    for (name, f) in stages
        first = @elapsed (f(); sync())
        times = [(@elapsed (f(); sync())) for _ in 1:REPS]
        @printf("  %-20s %10.1f %10.2f %10.2f\n", name, 1e3first, 1e3minimum(times), 1e3median(times))
    end
end

Base.invokelatest(profile, OPTIONS, BACKEND)
