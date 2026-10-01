# Dual-number seeds and kernel plumbing shared by the oracles. Distinct tags keep
# nested and sequential differentiations apart; every type is isbits, so the same
# code runs in KernelAbstractions kernels on the CPU and on CUDA devices.

struct TagPoint end    # 10 sigma-point inputs (state, control)
struct TagArc end      # 56 arc inputs (mean, feedforward, gain, factor)
struct TagFactor end   # 28 factor entries (Cholesky sensitivities)
struct TagOuter end    # pair-seeded second derivatives
struct TagInner end

const PointDual = ForwardDiff.Dual{TagPoint,Float64,10}

@inline unit_partials(::Val{N}, i) where {N} = ForwardDiff.Partials{N,Float64}(ntuple(j -> Float64(j == i), Val(N)))

"Seed `v` as coordinates `offset+1 : offset+length(v)` of an N-direction dual."
@inline seed(::Type{Tag}, v::SVector{L,Float64}, offset::Int, ::Val{N}) where {Tag,L,N} =
    SVector{L}(ntuple(i -> ForwardDiff.Dual{Tag}(v[i], unit_partials(Val(N), offset + i)), Val(L)))

"""
Seed `v` (coordinates `offset+1 : offset+length(v)`) on the chunk of `C` directions
`c0+1 : c0+C` only. Each partial derivative is computed by the same operations as
with the full-width `seed`, so a chunk reproduces its columns bitwise.
"""
@inline chunk_seed(::Type{Tag}, v::SVector{L,Float64}, offset::Int, c0::Int, ::Val{C}) where {Tag,L,C} =
    SVector{L}(ntuple(i -> ForwardDiff.Dual{Tag}(v[i], ForwardDiff.Partials{C,Float64}(
        ntuple(j -> Float64(c0 + j == offset + i), Val(C)))), Val(L)))
@inline chunk_width(::Val{C}) where {C} = C

@inline partial(d::ForwardDiff.Dual, j) = ForwardDiff.partials(d)[j]

"Scalar seeded along one outer and one inner direction (an entry of a Hessian)."
@inline function pair_seed(v::Float64, p_on::Bool, q_on::Bool)
    inner = ForwardDiff.Dual{TagInner}(v, ForwardDiff.Partials{1,Float64}((Float64(q_on),)))
    shift = ForwardDiff.Dual{TagInner}(Float64(p_on), ForwardDiff.Partials{1,Float64}((0.0,)))
    return ForwardDiff.Dual{TagOuter}(inner, ForwardDiff.Partials{1,typeof(inner)}((shift,)))
end
@inline pair_value(d) = ForwardDiff.partials(ForwardDiff.partials(d)[1])[1]
@inline single_seed(v::Float64, on::Bool) = ForwardDiff.Dual{TagInner}(v, ForwardDiff.Partials{1,Float64}((Float64(on),)))

"Position of (p, q), p >= q, in a row-major dense lower triangle."
@inline lower_position(p, q) = p * (p - 1) ÷ 2 + q

"""
Launch a kernel over `ndrange` items on the backend of its first argument. CPU launches
finish before returning. Device launches are asynchronous: kernels on one stream run in
order, and copies to the host synchronize.
"""
function launch!(kernel, ndrange, args...; workgroup::Int = 64)
    ndrange == 0 && return nothing
    backend = KernelAbstractions.get_backend(args[1])
    kernel(backend, workgroup)(args...; ndrange)
    backend isa KernelAbstractions.CPU && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function gather_kernel!(target, @Const(source), @Const(index))
    i = @index(Global, Linear)
    @inbounds target[i] = source[index[i]]
end

"target .= source[index] on any backend."
gather!(target, source, index) = launch!(gather_kernel!, length(target), target, source, index)
