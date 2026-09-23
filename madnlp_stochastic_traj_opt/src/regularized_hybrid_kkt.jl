# Loaded only with HybridKKT. Version 0.4 omits the dual diagonal in its
# condensed solve. This adapter retains its GPU factorization and Schur-CG
# algorithm while supporting MadNLP's regularization and restoration systems.

struct ShiftedHybridSchur{T,S,V}
    base::S
    shift::V
end
ShiftedHybridSchur(base,shift::V) where V = ShiftedHybridSchur{eltype(shift),typeof(base),V}(base,shift)
Base.size(S::ShiftedHybridSchur) = size(S.base)
Base.size(S::ShiftedHybridSchur,d::Int) = size(S.base)[d]
Base.eltype(::ShiftedHybridSchur{T}) where T = T
function LinearAlgebra.mul!(y::V,S::ShiftedHybridSchur{T},x::V,alpha::Number,beta::Number) where {T,V<:AbstractVector{T}}
    mul!(y,S.base,x,alpha,beta)
    y .+= alpha .* S.shift .* x
    return y
end

struct RegularizedHybridKKTSystem{T,VT,MT,QN,K,S} <: MadNLP.AbstractCondensedKKTSystem{T,VT,MT,QN}
    inner::K
    schur::S
    equality_scale::VT
    inequality_inverse::VT
    inequality_diagonal::VT
    effective_gamma::Base.RefValue{T}
end

@inline function Base.getproperty(kkt::RegularizedHybridKKTSystem,name::Symbol)
    if name in (:inner,:schur,:equality_scale,:inequality_inverse,:inequality_diagonal,:effective_gamma)
        return getfield(kkt,name)
    end
    return getproperty(getfield(kkt,:inner),name)
end
Base.propertynames(kkt::RegularizedHybridKKTSystem) =
    (fieldnames(typeof(kkt))...,propertynames(kkt.inner)...)

function MadNLP.create_kkt_system(::Type{RegularizedHybridKKTSystem},cb::MadNLP.SparseCallback{T,VT},linear_solver;kwargs...) where {T,VT}
    inner=MadNLP.create_kkt_system(HybridKKT.HybridCondensedKKTSystem,cb,linear_solver;kwargs...)
    eqscale=similar(inner.reg,length(inner.ind_eq)); fill!(eqscale,one(T))
    shift=similar(eqscale); fill!(shift,zero(T))
    inverse=similar(inner.reg,length(inner.ind_ineq)); fill!(inverse,one(T))
    diagonal=similar(inverse); fill!(diagonal,zero(T))
    schur=ShiftedHybridSchur(inner.S,shift)
    return RegularizedHybridKKTSystem{T,VT,typeof(inner.hess_com),typeof(inner.quasi_newton),typeof(inner),typeof(schur)}(
        inner,schur,eqscale,inverse,diagonal,Ref(inner.gamma[]))
end

MadNLP.initialize!(kkt::RegularizedHybridKKTSystem) = MadNLP.initialize!(kkt.inner)
MadNLP.compress_hessian!(kkt::RegularizedHybridKKTSystem) = MadNLP.compress_hessian!(kkt.inner)
MadNLP.compress_jacobian!(kkt::RegularizedHybridKKTSystem) = MadNLP.compress_jacobian!(kkt.inner)
MadNLP.jtprod!(y::AbstractVector,kkt::RegularizedHybridKKTSystem,x::AbstractVector) = MadNLP.jtprod!(y,kkt.inner,x)
MadNLP.is_inertia_correct(kkt::RegularizedHybridKKTSystem,p,z,n) = MadNLP.is_inertia_correct(kkt.inner,p,z,n)
# Like MadNLP's sparse condensed system, regularize the dual block during
# inertia correction. Its damping now appears in the actual linear system.
MadNLP.should_regularize_dual(::RegularizedHybridKKTSystem,p,z,n) = true
LinearAlgebra.mul!(w::MadNLP.AbstractKKTVector{T},kkt::RegularizedHybridKKTSystem{T},
    x::MadNLP.AbstractKKTVector{T},alpha,beta) where T = mul!(w,kkt.inner,x,alpha,beta)

function MadNLP.build_kkt!(kkt::RegularizedHybridKKTSystem{T}) where T
    n=size(kkt.hess_com,1)
    sigma_s=view(kkt.pr_diag,n+1:length(kkt.pr_diag))
    de=view(kkt.du_diag,kkt.ind_eq)
    di=view(kkt.du_diag,kkt.ind_ineq)
    # MadNLP's dual diagonal is nonpositive. Bound gamma so B=I+gamma*De
    # remains positive, which preserves a symmetric positive Schur operator.
    maximum(kkt.du_diag) <= zero(T) || error("unexpected positive dual diagonal")
    delta=isempty(de) ? zero(T) : maximum(abs,de)
    gamma=delta>0 ? min(kkt.gamma[],T(0.5)/delta) : kkt.gamma[]
    kkt.effective_gamma[]=gamma
    kkt.equality_scale .= one(T) .+ gamma .* de
    kkt.schur.shift .= .-de ./ kkt.equality_scale
    kkt.inequality_inverse .= one(T) ./ (one(T) .- di .* sigma_s)
    kkt.inequality_diagonal .= sigma_s .* kkt.inequality_inverse
    fill!(kkt.diag_buffer,zero(T))
    HybridKKT.index_copy!(kkt.diag_buffer,kkt.ind_ineq,kkt.inequality_diagonal)
    HybridKKT.fixed!(kkt.diag_buffer,kkt.ind_eq,gamma)
    MadNLP.build_condensed_aug_coord!(kkt.inner)
    return nothing
end

function MadNLP.solve_kkt!(kkt::RegularizedHybridKKTSystem{T},w::MadNLP.AbstractKKTVector) where T
    n,m=size(kkt.jt_csc)
    mi=length(kkt.ind_ineq)
    wx=MadNLP._madnlp_unsafe_wrap(MadNLP.full(w),n)
    ws=view(MadNLP.full(w),n+1:n+mi)
    wc=view(MadNLP.full(w),n+mi+1:n+mi+m)
    r1,vs,wz,wy=kkt.buffer3,kkt.buffer4,kkt.buffer5,kkt.buffer6
    G=kkt.G_csc
    HybridKKT.index_copy!(wy,wc,kkt.ind_eq)
    HybridKKT.index_copy!(wz,wc,kkt.ind_ineq)
    sigma_s=view(kkt.pr_diag,n+1:n+mi)
    di=view(kkt.du_diag,kkt.ind_ineq)
    MadNLP.reduce_rhs!(w.xp_lr,MadNLP.dual_lb(w),kkt.l_diag,w.xp_ur,MadNLP.dual_ub(w),kkt.u_diag)
    fill!(kkt.buffer1,zero(T))
    vs .= kkt.inequality_diagonal .* wz .+ kkt.inequality_inverse .* ws
    HybridKKT.index_copy!(kkt.buffer1,kkt.ind_ineq,vs)
    mul!(wx,kkt.jt_csc,kkt.buffer1,one(T),one(T))
    copyto!(r1,wx)
    mul!(r1,G',wy,kkt.effective_gamma[],one(T))
    copyto!(wx,r1)
    MadNLP.solve_linear_system!(kkt.linear_solver,r1)
    mul!(wy,G,r1,one(T),-one(T))
    HybridKKT.Krylov.krylov_solve!(kkt.iterative_linear_solver,kkt.schur,wy;
        atol=0.0,rtol=1e-10,verbose=0)
    copyto!(wy,kkt.iterative_linear_solver.x)
    mul!(wx,G',wy,-one(T),one(T))
    MadNLP.solve_linear_system!(kkt.linear_solver,wx)
    wy ./= kkt.equality_scale
    mul!(kkt.buffer2,kkt.jt_csc',wx)
    vj=view(kkt.buffer2,kkt.ind_ineq)
    copyto!(vs,ws)
    ws .= (vj .- wz .- di .* vs) .* kkt.inequality_inverse
    wz .= sigma_s .* ws .- vs
    HybridKKT.index_copy!(wc,kkt.ind_ineq,wz)
    HybridKKT.index_copy!(wc,kkt.ind_eq,wy)
    MadNLP.finish_aug_solve!(kkt,w)
    return w
end
