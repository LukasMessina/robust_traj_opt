# Row-major lower-triangular packing, identical to `LowerTriangular` of the JAX
# script: (1,1), (2,1), (2,2), (3,1), ... Entry (r, c), r >= c, is packed_index(r, c).

@inline packed_index(r, c) = r * (r - 1) ÷ 2 + c

const LTRI_ROWS = Tuple(r for r in 1:NX for c in 1:r)
const LTRI_COLS = Tuple(c for r in 1:NX for c in 1:r)
const LTRI_DIAGONAL = Tuple(packed_index(r, r) for r in 1:NX)
const MTRI_DIAGONAL = Tuple(packed_index(r, r) for r in 1:NP)

pack_lower(M::AbstractMatrix) = [M[r, c] for r in axes(M, 1) for c in 1:r]

function unpack_lower(entries::AbstractVector, n::Integer)
    M = zeros(eltype(entries), n, n)
    for r in 1:n, c in 1:r
        M[r, c] = entries[packed_index(r, c)]
    end
    return M
end

"Normalized node covariance (std .* L)(std .* L)' from packed factor entries."
function node_covariance(entries::AbstractVector, std::AbstractVector)
    F = unpack_lower(entries, NX) .* std
    return F * F'
end

"Static 7x7 lower-triangular matrix from 28 packed entries (generic element type)."
@inline function lower_matrix(v::SVector{28,T}) where {T}
    z = zero(T)
    return SMatrix{7,7,T}(ntuple(Val(49)) do linear
        r, c = (linear - 1) % 7 + 1, (linear - 1) ÷ 7 + 1
        r >= c ? v[packed_index(r, c)] : z
    end)
end

@inline function lower_matrix3(v::SVector{6,T}) where {T}
    z = zero(T)
    return SMatrix{3,3,T}(v[1], v[2], v[4], z, v[3], v[5], z, z, v[6])
end

"Column `j` of a packed 7x7 lower factor (zeros above the diagonal)."
@inline function packed_column(S::SVector{28,T}, j::Int) where {T}
    z = zero(T)
    return SVector{7,T}(ntuple(r -> r >= j ? S[packed_index(r, j)] : z, Val(7)))
end

"Packed Cholesky factor: the scalar recursion of `cholesky_lower` in the JAX script."
@generated function cholesky_packed(M::SMatrix{N,N,T}, pivot_floor) where {N,T}
    name(r, c) = Symbol(:F_, r, :_, c)
    body = Expr[]
    for r in 1:N, c in 1:r
        total = :(M[$r, $c])
        for k in 1:c-1
            total = :($total - $(name(r, k)) * $(name(c, k)))
        end
        push!(body, r == c ? :($(name(r, c)) = sqrt(max($total, pivot_floor))) :
                             :($(name(r, c)) = $total / $(name(c, c))))
    end
    entries = [name(r, c) for r in 1:N for c in 1:r]
    return quote
        Base.@_inline_meta
        $(body...)
        return SVector{$(length(entries))}($(entries...))
    end
end
