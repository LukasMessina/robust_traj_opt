# objective = :chance_sum: the thrust chance left-hand sides as the cost, weighted by the arc time steps,
#
#   J(x) = sum_k w_k (||S_k|| + psi rho(T_k)),   w_k = N dt_k / (t_f - t_0),
#
# with no epigraph variable. The term is evaluated by a second instance of the chance oracle
# (`chance_oracle`: per-row values, Jacobian and pair-seeded Hessians, CPU and GPU kernels) and added to the
# ExaModel by a wrapper: ExaModels' scalar objective oracles enter the gradient but not `hess_coord!`, and
# the exact Hessian is needed. The Hessian of the term is the oracle's with the multipliers sigma w_k
# (sigma the objective weight), appended to the model's Hessian structure. Each variable enters at most one
# chance row, so the gradient is a scatter without collisions.

struct ChanceObjectiveModel{T,VT,M<:NLPModels.AbstractNLPModel{T,VT},O,VI} <: NLPModels.AbstractNLPModel{T,VT}
    meta::NLPModels.NLPModelMeta{T,VT}
    counters::NLPModels.Counters
    base::M
    oracle::O
    weights::VT          # w_k, one per chance row
    entry_weights::VT    # w_k of every oracle Jacobian entry
    entry_cols::VI       # global variable of every oracle Jacobian entry
    values::VT
    jac::VT
    hess::VT
    multipliers::VT      # sigma w_k
    base_nnzh::Int
end

function ChanceObjectiveModel(base::NLPModels.AbstractNLPModel, oracle, weights::AbstractVector, backend)
    allunique(oracle.jac_cols) || error("a variable enters two chance rows: the gradient scatter would collide")
    device(v) = ExaModels.convert_array(v, backend)
    m = base.meta
    meta = NLPModels.NLPModelMeta(m.nvar; x0 = m.x0, lvar = m.lvar, uvar = m.uvar, ncon = m.ncon, y0 = m.y0,
        lcon = m.lcon, ucon = m.ucon, nnzj = m.nnzj, nnzh = m.nnzh + oracle.nnzh, lin = m.lin, minimize = m.minimize,
        name = m.name)
    w = Vector{Float64}(weights)
    return ChanceObjectiveModel(meta, NLPModels.Counters(), base, oracle, device(w), device(w[oracle.jac_rows]),
        device(oracle.jac_cols), device(zeros(oracle.ncon)), device(zeros(oracle.nnzj)), device(zeros(oracle.nnzh)),
        device(zeros(oracle.ncon)), m.nnzh)
end

function NLPModels.obj(m::ChanceObjectiveModel, x::AbstractVector)
    m.oracle.f!(m.values, x)
    return NLPModels.obj(m.base, x) + sum(m.weights .* m.values)
end

function NLPModels.grad!(m::ChanceObjectiveModel, x::AbstractVector, g::AbstractVector)
    NLPModels.grad!(m.base, x, g)
    m.oracle.jac!(m.jac, x)
    view(g, m.entry_cols) .+= m.entry_weights .* m.jac
    return g
end

NLPModels.cons_nln!(m::ChanceObjectiveModel, x::AbstractVector, c::AbstractVector) = NLPModels.cons_nln!(m.base, x, c)
NLPModels.jac_structure!(m::ChanceObjectiveModel, rows::AbstractVector, cols::AbstractVector) =
    NLPModels.jac_structure!(m.base, rows, cols)
NLPModels.jac_coord!(m::ChanceObjectiveModel, x::AbstractVector, vals::AbstractVector) = NLPModels.jac_coord!(m.base, x, vals)
NLPModels.jprod_nln!(m::ChanceObjectiveModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector) =
    NLPModels.jprod_nln!(m.base, x, v, Jv)
NLPModels.jtprod_nln!(m::ChanceObjectiveModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector) =
    NLPModels.jtprod_nln!(m.base, x, v, Jtv)

function NLPModels.hess_structure!(m::ChanceObjectiveModel, rows::AbstractVector, cols::AbstractVector)
    n = m.base_nnzh
    NLPModels.hess_structure!(m.base, view(rows, 1:n), view(cols, 1:n))
    copyto!(view(rows, n+1:length(rows)), m.oracle.hess_rows)
    copyto!(view(cols, n+1:length(cols)), m.oracle.hess_cols)
    return rows, cols
end

function NLPModels.hess_coord!(m::ChanceObjectiveModel, x::AbstractVector, y::AbstractVector, vals::AbstractVector;
        obj_weight = one(eltype(x)))
    n = m.base_nnzh
    NLPModels.hess_coord!(m.base, x, y, view(vals, 1:n); obj_weight)
    m.multipliers .= obj_weight .* m.weights
    m.oracle.hess!(m.hess, x, m.multipliers)
    copyto!(view(vals, n+1:length(vals)), m.hess)
    return vals
end

function NLPModels.hess_coord!(m::ChanceObjectiveModel, x::AbstractVector, vals::AbstractVector; obj_weight = one(eltype(x)))
    return NLPModels.hess_coord!(m, x, fill!(similar(x, m.meta.ncon), 0), vals; obj_weight)
end
