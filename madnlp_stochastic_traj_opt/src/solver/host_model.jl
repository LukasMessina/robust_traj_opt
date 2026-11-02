# Hybrid execution (execution_mode = :hybrid): the model is built and evaluated on the device, MadNLP and
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

"The model MadNLP solves: the device model itself, or its host view in the hybrid mode."
solver_model(o::Options, model) = o.execution_mode == :hybrid ? HostModel(model) : model
