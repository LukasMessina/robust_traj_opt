# Included inside DoubleIntegratorStochastic. Only the nested control chance
# margin uses an oracle; its variables remain ordinary ExaModels variables.
import ForwardDiff

struct ChanceFirstTag end
struct ChanceSecondTag end

"Control chance margin from the eight nonzero UT control offsets."
@inline function local_control_chance(x, scale, weight, psi, epsilon, smoothing)
    s11 = zero(x[1]); s22 = zero(x[1]); s12 = zero(x[1])
    for column in 1:NX
        u1 = zero(x[1]); u2 = zero(x[1])
        for row in column:NX
            l = x[10 + packed_index(row,column)]
            u1 += x[2 + 2(row-1) + 1] * l
            u2 += x[2 + 2(row-1) + 2] * l
        end
        u1 *= scale; u2 *= scale
        s11 += weight*(u1*u1 + (-u1)*(-u1))
        s22 += weight*(u2*u2 + (-u2)*(-u2))
        s12 += weight*(u1*u2 + (-u1)*(-u2))
    end
    return control_norm(x[1],x[2],epsilon) + psi*spectral_radius_sqrt(s11,s22,s12,smoothing)
end

@inline first_dual(value, seed) = ForwardDiff.Dual{ChanceFirstTag}(value, (seed,))
@inline second_dual(value, seed) = ForwardDiff.Dual{ChanceSecondTag}(value, (seed,))

@kernel function chance_value_kernel!(values, @Const(x), @Const(indices), parameters)
    k = @index(Global, Linear)
    local_x = ntuple(j -> @inbounds(x[indices[j,k]]), Val(20))
    @inbounds values[k] = local_control_chance(local_x,parameters...)
end

@kernel function chance_jacobian_kernel!(values, @Const(x), @Const(indices), parameters)
    index = @index(Global, Linear)
    k = (index-1) ÷ 20 + 1
    p = (index-1) % 20 + 1
    local_x = ntuple(j -> first_dual(@inbounds(x[indices[j,k]]),Float64(j==p)), Val(20))
    value = local_control_chance(local_x,parameters...)
    @inbounds values[index] = ForwardDiff.partials(value)[1]
end

@kernel function chance_hessian_kernel!(values, @Const(x), @Const(y), @Const(indices),
                                       @Const(pair_rows), @Const(pair_cols), parameters)
    index = @index(Global, Linear)
    k = (index-1) ÷ 210 + 1
    pair = (index-1) % 210 + 1
    p = @inbounds pair_rows[pair]
    q = @inbounds pair_cols[pair]
    local_x = ntuple(Val(20)) do j
        primal_value = @inbounds x[indices[j,k]]
        # The outer derivative is p and the inner derivative is q.
        inner = first_dual(primal_value,Float64(j==q))
        seed = first_dual(Float64(j==p),0.0)
        second_dual(inner,seed)
    end
    value = local_control_chance(local_x,parameters...)
    @inbounds values[index] = y[k]*ForwardDiff.partials(ForwardDiff.partials(value)[1])[1]
end

function launch_chance!(kernel, output, args...)
    backend = KernelAbstractions.get_backend(output)
    kernel(backend,64)(output,args...;ndrange=length(output))
    KernelAbstractions.synchronize(backend)
    return nothing
end

function add_control_chance_oracle(c, o::Options, U, K, L, backend)
    N = o.n_arcs
    indices = Matrix{Int}(undef,20,N)
    for k in 1:N
        indices[:,k] = [U.offset .+ (2(k-1) .+ (1:2));
                        K.offset .+ (8(k-1) .+ (1:8));
                        L.offset .+ (10(k-1) .+ (1:10))]
    end
    pair_rows = [r for r in 1:20 for col in 1:r]
    pair_cols = [col for r in 1:20 for col in 1:r]
    jac_rows = repeat(collect(1:N);inner=20)
    jac_cols = vec(indices)
    hess_rows = [indices[r,k] for k in 1:N for r in 1:20 for col in 1:r]
    hess_cols = [indices[col,k] for k in 1:N for r in 1:20 for col in 1:r]
    # All local indices are ordered by the original variable blocks.
    @assert all(hess_rows .>= hess_cols)
    device_indices = ExaModels.convert_array(indices,backend)
    device_rows = ExaModels.convert_array(pair_rows,backend)
    device_cols = ExaModels.convert_array(pair_cols,backend)
    scale = NAUG+o.scaling_parameter
    parameters = (sqrt(scale),1/(2scale),sqrt(quantile(Chisq(NU),o.control_confidence)),
        o.control_norm_epsilon,o.spectral_eigenvalue_smoothing)
    oracle = ExaModels.VectorNonlinearOracle(
        nvar=24N+24, ncon=N, jac_rows=jac_rows, jac_cols=jac_cols,
        hess_rows=hess_rows, hess_cols=hess_cols,
        lcon=fill(-Inf,N), ucon=fill(o.u_max,N), adapt=Val(false),
        f! = (out,x) -> launch_chance!(chance_value_kernel!,out,x,device_indices,parameters),
        jac! = (out,x) -> launch_chance!(chance_jacobian_kernel!,out,x,device_indices,parameters),
        hess! = (out,x,y) -> launch_chance!(chance_hessian_kernel!,out,x,y,device_indices,device_rows,device_cols,parameters))
    return ExaModels.constraint(c,oracle)
end
