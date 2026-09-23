"A scalar covariance entry, traced at model construction with fixed r and c."
function covariance_entry(L, k, r::Int, c::Int)
    return sum(L[packed_index(r, j), k] * L[packed_index(c, j), k] for j in 1:min(r, c))
end

function build_stochastic_model(o::Options, Lc, seed, backend)
    N, h = o.n_arcs, o.dt
    c = ExaCore(Float64; backend)
    @add_var(c, means, 1:NX, 1:N+1; start=seed.means)
    @add_var(c, feedforward, 1:NU, 1:N; start=seed.feedforward)
    @add_var(c, gains, 1:NU, 1:NX, 1:N; start=seed.gains)
    @add_var(c, cholesky_factor, 1:10, 1:N+1; start=seed.cholesky_factor)
    @add_var(c, terminal_margin, 1:10; start=seed.terminal_margin)
    L = cholesky_factor
    Z, weights = unscented_rule(o)
    @add_par(c, z, Z[1:4, :])
    @add_par(c, w, weights)

    # Packed L is already a square root: its columns generate precisely the
    # symmetric sigma set obtained from cholesky((12+kappa)*L*L').
    @add_expr(c, dx1, L[1,k]*z[1,s] for k in 1:N, s in 1:NSIGMA)
    @add_expr(c, dx2, L[2,k]*z[1,s] + L[3,k]*z[2,s] for k in 1:N, s in 1:NSIGMA)
    @add_expr(c, dx3, L[4,k]*z[1,s] + L[5,k]*z[2,s] + L[6,k]*z[3,s] for k in 1:N, s in 1:NSIGMA)
    @add_expr(c, dx4, L[7,k]*z[1,s] + L[8,k]*z[2,s] + L[9,k]*z[3,s] + L[10,k]*z[4,s] for k in 1:N, s in 1:NSIGMA)
    @add_expr(c, delta_u, gains[i,1,k]*dx1[k,s] + gains[i,2,k]*dx2[k,s] +
        gains[i,3,k]*dx3[k,s] + gains[i,4,k]*dx4[k,s] for i in 1:NU, k in 1:N, s in 1:NSIGMA)
    # Center the sigma cloud before propagation. The affine ShARK map sends
    # symmetric +/- offsets to symmetric +/- output offsets, so their weighted
    # mean is exactly zero. This is the same centered UT, evaluated without
    # subtracting nearly equal absolute positions or duplicating a 25-term mean
    # expression inside every covariance term. No F*P*F' recurrence is used.
    moment_rows = Any[]

    # Accumulate all 25 sigma contributions into each mean matching row.
    for i in 1:NX
        if i <= NU
            @add_con(c, mean_defect, means[i,k+1] - shark_position(means[i,k], means[i+2,k],
                feedforward[i,k], 0.0, 0.0, 0.0, h) for k in 1:N)
        else
            @add_con(c, mean_defect, means[i,k+1] - shark_velocity(means[i,k],
                feedforward[i-2,k], 0.0, h) for k in 1:N)
        end
        push!(moment_rows,mean_defect)
    end
    covariance = Matrix{Any}(undef, NX, NX)
    for (r, col) in LOWER
        @add_expr(c, Pij, covariance_entry(L, k, r, col) for k in 1:N+1)
        covariance[r,col] = covariance[col,r] = Pij
        @add_con(c, defects, Pij[k+1] for k in 1:N)
        push!(moment_rows,defects)
        target = o.x0_covariance[r,col]
        @add_con(c, Pij[1] - target for dummy in 1:1)
        di = 1 / sqrt(o.xf_covariance[r,r])
        dj = 1 / sqrt(o.xf_covariance[col,col])
        identity_entry = Float64(r == col)
        @add_con(c, identity_entry - di*dj*Pij[N+1] -
            sum(terminal_margin[packed_index(r,j)] * terminal_margin[packed_index(col,j)] for j in 1:col)
            for dummy in 1:1)
    end

    c = add_moment_evaluator(c,o,moment_rows,gains,L,Lc,backend)
    o.print_level > 1 && println("  Mean and covariance constraints ready")
    # The center and the 16 noise-only points have delta_u == 0. Their
    # covariance contribution is exactly zero, for any kappa. Retain the eight
    # state-offset points (both signs); evaluate their weighted outer products.
    epsilon = o.control_norm_epsilon
    phi = quantile(Normal(), o.path_confidence)
    @add_expr(c, norms, control_norm(feedforward[1,k], feedforward[2,k], epsilon) for k in 1:N)
    c = add_control_chance_oracle(c,o,feedforward,gains,L,backend)
    @add_obj(c, h*norms[k] for k in 1:N)
    r11, r12, r22 = o.R[1,1], o.R[1,2], o.R[2,2]
    @add_obj(c, h*w[s]*(r11*delta_u[1,k,s]^2 + 2r12*delta_u[1,k,s]*delta_u[2,k,s] +
        r22*delta_u[2,k,s]^2) for k in 1:N, s in (2,3,4,5,14,15,16,17))
    for (r, col) in LOWER
        Pij = covariance[r,col]
        coefficient = h * o.Q[r,col] * (r == col ? 1 : 2)
        iszero(coefficient) && continue
        @add_obj(c, coefficient*Pij[k] for k in 1:N)
    end
    for (a, b) in ((o.a1, o.b1), (o.a2, o.b2))
        # a' P a = ||L' a||^2: numerically nonnegative, identical to Python.
        a1, a2, a3, a4 = a
        @add_con(c, a1*means[1,k]+a2*means[2,k]+a3*means[3,k]+a4*means[4,k] + phi*sqrt(
            (a1*L[1,k]+a2*L[2,k]+a3*L[4,k]+a4*L[7,k])^2 +
            (a2*L[3,k]+a3*L[5,k]+a4*L[8,k])^2 +
            (a3*L[6,k]+a4*L[9,k])^2 + (a4*L[10,k])^2)
            for k in 1:N; lcon=-Inf, ucon=b)
    end
    @add_con(c, means[i,1] - target for (i,target) in collect(enumerate(o.x0_mean)))
    @add_con(c, means[i,N+1] - target for (i,target) in collect(enumerate(o.xf_mean)))
    @add_con(c, L[d,k] for d in DIAGONAL, k in 1:N+1; lcon=0.0, ucon=Inf)
    @add_con(c, terminal_margin[d] for d in DIAGONAL; lcon=o.terminal_margin_floor, ucon=Inf)
    o.print_level > 1 && println("  Constructing sparse ExaModel derivatives")
    model = ExaModel(c)
    @assert model.meta.nvar == 24N + 24 "extra model variables were introduced"
    @assert model.meta.ncon == 21N + 36
    @assert count(model.meta.lcon .== model.meta.ucon) == 14N + 28
    return (; model, means, feedforward, gains, cholesky_factor, terminal_margin)
end
