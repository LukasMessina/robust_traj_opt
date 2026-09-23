"Minimum-energy transfer with the same nominal path constraints as Python."
function build_nominal_model(o::Options, backend)
    N, h = o.n_arcs, o.dt
    c = ExaCore(Float64; backend)
    xseed = hcat(((1 - t) * o.x0_mean + t * o.xf_mean for t in range(0, 1; length=N+1))...)
    @add_var(c, means, 1:NX, 1:N+1; start=xseed)
    @add_var(c, feedforward, 1:NU, 1:N; start=0.1)
    @add_obj(c, h * feedforward[i, k]^2 for i in 1:NU, k in 1:N)
    @add_con(c, means[i, 1] - target for (i, target) in collect(enumerate(o.x0_mean)))
    @add_con(c, means[i, N+1] - target for (i, target) in collect(enumerate(o.xf_mean)))
    # The zero-noise ShARK stages are exact for this ZOH double integrator,
    # as is Python's deterministic Tsit5 step.
    @add_con(c, means[i, k+1] - shark_position(means[i, k], means[i+2, k],
        feedforward[i, k], 0.0, 0.0, 0.0, h) for i in 1:NU, k in 1:N)
    @add_con(c, means[i+2, k+1] - shark_velocity(means[i+2, k],
        feedforward[i, k], 0.0, h) for i in 1:NU, k in 1:N)
    for (a, b) in ((o.a1, o.b1), (o.a2, o.b2))
        a1, a2, a3, a4 = a
        @add_con(c, a1 * means[1, k] + a2 * means[2, k] + a3 * means[3, k] +
            a4 * means[4, k] for k in 1:N; lcon=-Inf, ucon=b)
    end
    model = ExaModel(c)
    @assert model.meta.nvar == 6N + 4
    return (; model, means, feedforward)
end

function tvlqr_gains(o::Options, A, B)
    Q = diagm(1 ./ diag(o.x0_covariance))
    R = o.tvlqr_control_weight_scale * Matrix{Float64}(I, NU, NU) / o.u_max^2
    V = diagm(1 ./ diag(o.xf_covariance))
    K = zeros(NU, NX, o.n_arcs)
    for k in o.n_arcs:-1:1
        gain = (R + B' * V * B) \ (B' * V * A)
        K[:, :, k] = -gain
        F = A - B * gain
        V = Q + gain' * R * gain + F' * V * F
        V = (V + V') / 2
    end
    return K
end

function build_seed(o::Options, A, B, Lc, nominal)
    N = o.n_arcs
    K = if o.initial_gain_uniform !== nothing
        fill(o.initial_gain_uniform,NU,NX,N)
    else
        o.warm_start_gains ? tvlqr_gains(o, A, B) : zeros(NU, NX, N)
    end
    if o.initial_gain_max_abs !== nothing
        peak = maximum(abs,K)
        peak > 0 && (K .*= o.initial_gain_max_abs / peak)
    end
    means, U = copy(nominal.means), copy(nominal.feedforward)
    means[:, 1] = o.x0_mean
    factors = zeros(10, N+1)
    L = Matrix(cholesky(Symmetric(o.x0_covariance)).L)
    factors[:, 1] = pack_lower(L)
    P = o.x0_covariance
    for k in 1:N
        means[:, k+1], P, _ = propagation_arc(o, Lc, means[:, k], L, U[:, k], K[:, :, k])
        L = Matrix(cholesky(Symmetric(P)).L)
        factors[:, k+1] = pack_lower(L)
    end
    D = diagm(1 ./ sqrt.(diag(o.xf_covariance)))
    headroom = Matrix{Float64}(I, NX, NX) - D * P * D
    E = eigen(Symmetric(headroom))
    # Same PSD projection and squared floor as the reference seed. This does
    # not assert terminal feasibility when a modified problem has negative slack.
    margin = cholesky(Symmetric(E.vectors * Diagonal(max.(E.values, 0.0)) * E.vectors' +
        o.terminal_margin_floor^2 * I)).L
    return (; means, feedforward=U, gains=K, cholesky_factor=factors, terminal_margin=pack_lower(margin))
end
