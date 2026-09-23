function original_violation(model, x)
    values = Array(NLPModels.cons(model, x))
    lower, upper = Array(model.meta.lcon), Array(model.meta.ucon)
    xv = Array(x)
    return max(maximum(max.(lower - values, values - upper, 0.0)),
        maximum(max.(Array(model.meta.lvar) - xv, xv - Array(model.meta.uvar), 0.0)))
end

function extract_solution(stats, blocks)
    get(name) = Array(ExaModels.solution(stats, getproperty(blocks, name)))
    means, U, K = get(:means), get(:feedforward), get(:gains)
    factors, margin = get(:cholesky_factor), get(:terminal_margin)
    P = cat((let L=unpack_lower(factors[:,k]); L*L' end for k in axes(factors,2))...; dims=3)
    return (; means, feedforward=U, gains=K, cholesky_factor=factors, terminal_margin=margin, state_covariances=P)
end

function diagnostics(o::Options, Lc, sol)
    S = zeros(NU, NU, o.n_arcs)
    mean_defect, covariance_defect = 0.0, 0.0
    for k in 1:o.n_arcs
        mu, P, Sk = propagation_arc(o, Lc, sol.means[:,k], unpack_lower(sol.cholesky_factor[:,k]),
            sol.feedforward[:,k], sol.gains[:,:,k])
        S[:,:,k] = Sk
        mean_defect = max(mean_defect, maximum(abs, mu-sol.means[:,k+1]))
        covariance_defect = max(covariance_defect, maximum(abs, P-sol.state_covariances[:,:,k+1]))
    end
    exact_radii = [sqrt(max(eigmax(Symmetric(S[:,:,k])),0.0)) for k in 1:o.n_arcs]
    radii = [spectral_radius_sqrt(S[1,1,k],S[2,2,k],S[1,2,k],o.spectral_eigenvalue_smoothing) for k in 1:o.n_arcs]
    norms = [control_norm(sol.feedforward[1,k], sol.feedforward[2,k], o.control_norm_epsilon) for k in 1:o.n_arcs]
    psi = sqrt(quantile(Chisq(NU),o.control_confidence))
    phi = quantile(Normal(),o.path_confidence)
    path_values = zeros(2,o.n_arcs)
    for (j,(a,b)) in enumerate(((o.a1,o.b1),(o.a2,o.b2)))
        for k in 1:o.n_arcs
            path_values[j,k] = dot(a,sol.means[:,k]) + phi*sqrt(max(dot(a,sol.state_covariances[:,:,k]*a),0.0)) - b
        end
    end
    D = diagm(1 ./ sqrt.(diag(o.xf_covariance)))
    headroom = Matrix{Float64}(I,NX,NX)-D*sol.state_covariances[:,:,end]*D
    M = unpack_lower(sol.terminal_margin)
    terminal_residual = maximum(abs,headroom-M*M')
    independent_violation = max(mean_defect,covariance_defect,terminal_residual,
        maximum(abs,sol.means[:,1]-o.x0_mean),maximum(abs,sol.means[:,end]-o.xf_mean),
        maximum(abs,sol.state_covariances[:,:,1]-o.x0_covariance),
        maximum(path_values),maximum(norms+psi*radii)-o.u_max,
        -minimum(sol.cholesky_factor[DIAGONAL,:]),o.terminal_margin_floor-minimum(diag(M)))
    objective = o.dt * sum(norms[k]+tr(o.Q*sol.state_covariances[:,:,k])+tr(o.R*S[:,:,k]) for k in 1:o.n_arcs)
    return (; control_covariances=S, radii, control_chance=norms+psi*radii, objective,
        mean_defect, covariance_defect, smoothing_bias=maximum(radii-exact_radii),
        independent_violation, terminal_residual, terminal_headroom_min_eigenvalue=eigmin(Symmetric(headroom)))
end
