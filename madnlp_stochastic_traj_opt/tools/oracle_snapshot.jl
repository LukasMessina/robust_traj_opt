# Bitwise regression check of the NLP evaluations. `save` stores the constraint,
# Jacobian and Lagrangian Hessian values of the full model, and the three UT oracle
# callbacks, at three fixed points; `compare` recomputes them with the current code and
# reports, per quantity, whether they are bitwise identical and the largest difference.
#   julia --project=madnlp_stochastic_traj_opt --threads=auto madnlp_stochastic_traj_opt/tools/oracle_snapshot.jl save|compare cpu|gpu CASE FILE
include(joinpath(@__DIR__, "..", "src", "CR3BPStochastic.jl"))
using .CR3BPStochastic
using Printf, Random, Serialization
import NLPModels
const C = CR3BPStochastic

const ACTION, MODE, CASE, FILE = ARGS[1], ARGS[2], ARGS[3], ARGS[4]
const OPTIONS = C.validate_options(C.Options(; case_id = CASE, execution_mode = Symbol(MODE)))
const BACKEND = C.execution_backend(OPTIONS)

"Every quantity at the fixed points."
function evaluate(o, backend)
    case = C.case_data(o.case_id)
    normalization = C.build_normalization(case)
    p = C.ut_params(case, o, normalization)
    ref = C.load_reference(case, o)
    seed = C.build_initial_guess(case, o, normalization, p, ref)
    scaling = C.covariance_scaling(seed, normalization)
    blocks = C.build_stochastic_model(case, o, ref, seed, scaling, p, backend)
    model, ut, n = blocks.model, blocks.ut, C.n_arcs(ref)
    device(v) = copyto!(similar(model.meta.x0, length(v)), v)
    rng = Xoshiro(2024)
    x0 = Array(model.meta.x0)
    points = [x0, x0 .* (1 .+ 1e-3 .* randn(rng, length(x0))), x0 .* (1 .+ 1e-2 .* randn(rng, length(x0)))]
    y = randn(rng, model.meta.ncon)
    yl = y[C.NX+C.NP+1:C.NX+C.NP+35n]            # multipliers of the UT rows
    results = Dict{String,Vector{Float64}}()
    for (k, xh) in enumerate(points)
        x = device(xh)
        xl = device(xh[1:model.meta.nvar-C.NM])   # the UT oracle's local vector (all but the margin)
        results["model constraints @$k"] = Array(NLPModels.cons(model, x))
        results["model Jacobian @$k"] = Array(NLPModels.jac_coord(model, x))
        results["model Hessian @$k"] = Array(NLPModels.hess_coord(model, x, device(y)))
        out = similar(x, 35n)
        C.ut_values!(out, xl, ut, p, n)
        results["UT values @$k"] = Array(out)
        vals = similar(x, length(ut.structure.jac_rows))
        C.ut_jacobian!(vals, xl, ut, p, n)
        results["UT Jacobian @$k"] = Array(vals)
        vals = similar(x, length(ut.structure.hess_rows))
        C.ut_hessian!(vals, xl, device(yl), ut, p, n)
        results["UT Hessian @$k"] = Array(vals)
    end
    return results
end

function compare(current, reference)
    identical = true
    @printf("  %-26s %8s %12s %12s\n", "quantity", "bitwise", "max |diff|", "max rel")
    for key in sort(collect(keys(reference)))
        a, b = current[key], reference[key]
        same = length(a) == length(b) && all(i -> a[i] === b[i], eachindex(a))   # bit patterns
        diff = length(a) == length(b) ? maximum(abs, a - b; init = 0.0) : Inf
        rel = diff / max(maximum(abs, b; init = 0.0), 1e-300)
        identical &= same
        @printf("  %-26s %8s %12.3e %12.3e\n", key, same ? "yes" : "NO", diff, rel)
    end
    return identical
end

function main()
    current = Base.invokelatest(evaluate, OPTIONS, BACKEND)
    if ACTION == "save"
        serialize(FILE, current)
        println("saved $(length(current)) quantities to $FILE")
    else
        println("$MODE $CASE vs $(basename(FILE))")
        println(compare(current, deserialize(FILE)) ? "ALL BITWISE IDENTICAL" : "DIFFERS")
    end
end

main()
