# Shared helpers of the test files (loaded by runtests.jl after the module).
using Test, LinearAlgebra, Random, SparseArrays, StaticArrays
import ForwardDiff
import NLPModels
const C = CR3BPStochastic

"A truncated problem (default 3 arcs) built from the resampled reference, without re-optimization."
function truncated_setup(; arcs = 3, backend = nothing, options...)
    o = C.Options(; truncated_uniform_mesh_arcs = arcs, options...)
    case = C.case_data(o.case_id)
    normalization = C.build_normalization(case)
    p = C.ut_params(case, o, normalization)
    ref = C.load_reference(case, o)
    seed = C.build_initial_guess(case, o, normalization, p, ref)
    scaling = C.covariance_scaling(seed, normalization)
    blocks = C.build_stochastic_model(case, o, ref, seed, scaling, p, backend)
    return (; o, case, normalization, p, ref, seed, scaling, blocks)
end

"Structure arrays live on the model's device; copy them to the host."
function sparsity(model, jacobian::Bool)
    nnz = jacobian ? model.meta.nnzj : model.meta.nnzh
    rows, cols = similar(model.meta.x0, Int, nnz), similar(model.meta.x0, Int, nnz)
    jacobian ? NLPModels.jac_structure!(model, rows, cols) : NLPModels.hess_structure!(model, rows, cols)
    return Array(rows), Array(cols)
end

function dense_jacobian(model, x)
    rows, cols = sparsity(model, true)
    return Matrix(sparse(rows, cols, Array(NLPModels.jac_coord(model, x)), model.meta.ncon, model.meta.nvar))
end

lagrangian_gradient(model, x, y) = Array(NLPModels.grad(model, x)) + dense_jacobian(model, x)' * Array(y)

function dense_hessian(model, x, y)
    rows, cols = sparsity(model, false)
    H = Matrix(sparse(rows, cols, Array(NLPModels.hess_coord(model, x, y)), model.meta.nvar, model.meta.nvar))
    return H + H' - Diagonal(diag(H))
end

"Central differences of a vector function, one column per coordinate of x."
central_differences(f, x; delta = 1e-6) =
    reduce(hcat, [(f(x + delta * e) - f(x - delta * e)) / (2delta) for e in eachcol(Matrix{Float64}(I, length(x), length(x)))])
