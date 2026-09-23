# Included inside DoubleIntegratorStochastic. The UT is a loop over the same
# 25 sigma points on CPU and GPU. Sparse AD callbacks avoid exponentially large
# symbolic expression types and do not introduce auxiliary decision variables.
const LOWER_TUPLE = Tuple(LOWER)

@inline function accumulate_moments(previous,y,weight)
    return ntuple(Val(14)) do j
        if j <= 4
            previous[j]+weight*y[j]
        else
            r,c = LOWER_TUPLE[j-4]
            previous[j]+weight*y[r]*y[c]
        end
    end
end

@inline function center_moments(accum)
    return ntuple(Val(14)) do j
        if j <= 4
            -accum[j]
        else
            r,c = LOWER_TUPLE[j-4]
            -(accum[j]-accum[r]*accum[c])
        end
    end
end

@inline function local_ut_moments(x, z, gw, gh, weights, h)
    accum = ntuple(_ -> zero(x[1]), Val(14))
    for s in 1:NSIGMA
        dx = ntuple(Val(4)) do r
            value = zero(x[1])
            for col in 1:r
                value += x[8+packed_index(r,col)]*z[col,s]
            end
            value
        end
        u1 = x[1]*dx[1]+x[3]*dx[2]+x[5]*dx[3]+x[7]*dx[4]
        u2 = x[2]*dx[1]+x[4]*dx[2]+x[6]*dx[3]+x[8]*dx[4]
        y = (shark_position(dx[1],dx[3],u1,gw[1,s],gw[3,s],gh[3,s],h),
             shark_position(dx[2],dx[4],u2,gw[2,s],gw[4,s],gh[4,s],h),
             shark_velocity(dx[3],u1,gw[3,s],h),
             shark_velocity(dx[4],u2,gw[4,s],h))
        accum = accumulate_moments(accum,y,weights[s])
    end
    return center_moments(accum)
end

@kernel function moment_value_kernel!(out,@Const(x),@Const(indices),z,gw,gh,w,h,N)
    k = @index(Global,Linear)
    local_x = ntuple(j -> @inbounds(x[indices[j,k]]),Val(18))
    values = local_ut_moments(local_x,z,gw,gh,w,h)
    for r in 1:14
        @inbounds out[(r-1)*N+k] = values[r]
    end
end

@kernel function moment_jacobian_kernel!(out,@Const(x),@Const(indices),z,gw,gh,w,h)
    index = @index(Global,Linear)
    k,p = (index-1) ÷ 18+1, (index-1)%18+1
    local_x = ntuple(j -> first_dual(@inbounds(x[indices[j,k]]),Float64(j==p)),Val(18))
    values = local_ut_moments(local_x,z,gw,gh,w,h)
    for r in 1:14
        @inbounds out[(index-1)*14+r] = ForwardDiff.partials(values[r])[1]
    end
end

@kernel function moment_hessian_kernel!(out,@Const(x),@Const(y),@Const(indices),
        @Const(pair_rows),@Const(pair_cols),z,gw,gh,w,h,N)
    index = @index(Global,Linear)
    k,pair = (index-1) ÷ 171+1, (index-1)%171+1
    p,q = @inbounds(pair_rows[pair]), @inbounds(pair_cols[pair])
    local_x = ntuple(Val(18)) do j
        value = @inbounds x[indices[j,k]]
        second_dual(first_dual(value,Float64(j==q)),first_dual(Float64(j==p),0.0))
    end
    values = local_ut_moments(local_x,z,gw,gh,w,h)
    result = zero(eltype(out))
    for r in 1:14
        result += @inbounds(y[(r-1)*N+k])*ForwardDiff.partials(ForwardDiff.partials(values[r])[1])[1]
    end
    @inbounds out[index] = result
end

function launch_moment!(kernel,out,ndrange,args...)
    backend = KernelAbstractions.get_backend(out)
    kernel(backend,64)(out,args...;ndrange)
    KernelAbstractions.synchronize(backend)
    return nothing
end

function add_moment_evaluator(c,o,rows,K,L,Lc,backend)
    N = o.n_arcs
    # add_eval supplies the concatenated (K,L) blocks as its local vector.
    indices = hcat(([8(k-1) .+ (1:8); 8N+10(k-1) .+ (1:10)] for k in 1:N)...)
    jr = [(r-1)*N+k for k in 1:N for p in 1:18 for r in 1:14]
    jc = [indices[p,k] for k in 1:N for p in 1:18 for r in 1:14]
    pr = [r for r in 1:18 for col in 1:r]
    pc = [col for r in 1:18 for col in 1:r]
    hr = [indices[r,k] for k in 1:N for r in 1:18 for col in 1:r]
    hc = [indices[col,k] for k in 1:N for r in 1:18 for col in 1:r]
    cv(a) = ExaModels.convert_array(a,backend)
    Z,w = unscented_rule(o)
    iz, ir, ic = cv(indices),cv(pr),cv(pc)
    z,gw,gh,weights = cv(Z[1:4,:]),cv(Lc*(sqrt(o.dt)*Z[5:8,:])),
        cv(Lc*(sqrt(o.dt/12)*Z[9:12,:])),cv(w)
    c,_ = ExaModels.add_eval(c,Tuple(rows),(K,L),
        (out,x) -> launch_moment!(moment_value_kernel!,out,N,x,iz,z,gw,gh,weights,o.dt,N);
        jac! = (out,x) -> launch_moment!(moment_jacobian_kernel!,out,18N,x,iz,z,gw,gh,weights,o.dt),
        hess! = (out,x,y) -> launch_moment!(moment_hessian_kernel!,out,171N,x,y,iz,ir,ic,z,gw,gh,weights,o.dt,N),
        jac_structure! = (r,c) -> (append!(r,jr);append!(c,jc)),
        hess_structure! = (r,c) -> (append!(r,hr);append!(c,hc)),adapt=Val(false))
    return c
end
