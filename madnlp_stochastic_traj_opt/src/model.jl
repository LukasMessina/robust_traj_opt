# Stochastic multiple-shooting NLP (the seeded TrajectoryNLP of the JAX script),
# built from a `StochasticLayout` shared by the single-phase cases and the
# multiphase Psyche case.
#
# Arc k maps node k to node k+1 (n arcs, n + 1 nodes). Variables: node means 7(n+1),
# normalized feedforward 3C and scaled gains K_hat = K_tilde / gain_scale 18C of the
# C controlled arcs, seed-scaled node Cholesky factors of nodes 2..n+1 28n, terminal
# margin factor 21. Constraints: fixed node components (boundary rows), mean defects 7n,
# covariance defects 28n, terminal residual 21, thrust chance C, then case-specific rows.
# Bounds: node-mean bounds of the layout, nonnegative factor diagonals, margin diagonal
# >= terminal_margin_floor, scaled gains |K_hat| <= gain_bound.
# A single-phase case has C = n, unit arc weights and boundary rows x(1) = x0 (7) and
# x(n+1) = target (6): 56 n + 28 variables and 36 n + 34 constraints.
# chance_formulation = :slack with polar or throttle controls (any layout) adds tau_hat (nc) and G_hat (6nc) and the
# rows of `chance_slack_oracle`; the chance rows are Gamma_k + psi s_k tau_hat_k <= 1 and the chance_sum cost
# sum_k w_k (Gamma_k + psi s_k tau_hat_k). With cartesian controls (single-phase only) it adds tau_hat (n) and the packed 3x3
# factor G_hat (6n), the chance row becomes ||S_k||_eps + psi s_k tau_hat_k <= 1, and 6n
# rows tau_hat_k^2 I - T_k / s_k^2 - G_hat_k G_hat_k' = 0 follow it (s_k = sqrt(lambda_max(T_k))
# at the seed); 63 n + 28 variables and 42 n + 34 constraints.
#
# Objective with arc weights w_k (w_P = state_covariance_weight, w = control_covariance_weight):
#   sum_controlled w_k (||S_k||_eps + w tr(T_k)) + sum_all w_k w_P tr(P_k[1:6,1:6]) / c^2,
# P_k the covariance at the start node of arc k, with tr(T_k) in the closed form of
# `control_covariance`:
#   tr(T) = (1 + s2 + 2 s4) ct tr(K M K') + (s2 + 2 s4) ||f||^2 + 3 eps,
#   tr(K M K') = d tr(K P6 K') + d sum_qb K_qb^2 R_nav,b.
# Terms constant in the variables (the fixed initial covariance and 3 w eps per controlled
# arc) are left out of the NLP and returned as `objective_constant`.
# objective = :quantile replaces the cost by the weighted sum of the chance left-hand sides (epigraph variables);
# objective = :chance_sum uses them directly, sum_k w_k (||S_k|| + psi rho(T_k)) through `ChanceObjectiveModel`.
# norm_epigraph replaces ||S_k||_eps by variables s_k >= 0 with rows s_k^2 - ||S_k||^2 >= 0 (after the
# chance rows) in the objective and the chance rows (s_k + psi rho(T_k) <= 1): no 1/eps curvature at coast arcs.
# control_parameterization = :polar makes the ff block (Gamma_k, alpha_k, beta_k), S_k = Gamma_k d_k with
# Gamma_k >= 0 and a unit d_k (`polar_direction`): ||S_k|| = Gamma_k in the objective and the chance rows,
# ||S_k||^2 = Gamma_k^2 in tr(T_k), and the UT oracles map the inputs (no eps anywhere in the model).
# control_parameterization = :throttle (the deterministic optimisation's controls) makes the ff block the
# direction d_k with rows ||d_k||^2 - 1 = 0 (after the defect rows) and adds the throttle Gamma_k in [0, 1]
# right after the factors (an input of the UT and chance oracles): ||S_k|| = Gamma_k, Gamma_k^2 ||d_k||^2 in tr(T_k).
# sigma_command_lifting (throttle only) adds, after the throttle, (eta, e) of the NLIFT = 24 outer sigma points
# that carry feedback on every controlled arc (`lift_index`): the UT oracles take c_o = eta_o e_o in place of
# Gamma d + sigma K m_o (K_hat is then not an oracle input), and rows Gamma_k d_k + sigma gs K_hat m_o - eta_o e_o = 0
# and ||e_o||^2 = 1 (after the unit-direction rows) tie them; eta_o is free (the sign is a gauge). No 1/||c_o||
# curvature on coast arcs: the mass flow and the Gates factors see |eta_o| times smooth functions of the unit e_o.
# covariance_defect = :factor or :qr writes the covariance rows in square-root form L_{k+1} - R_k,
# R_k R_k' = V_k (`ut_factor.jl`); the native part is then linear.
# terminal_covariance = :cost drops the margin factor and the terminal rows and adds the Bryson
# term w_N tr(G P_N G')/c^2 to the standard cost.
# objective = :terminal_dispersion minimizes one more variable, the level t of the terminal rows
# t I - G P_N G' - M M' = 0 (the margin starts at chol(t0 I - G P_N G'), t0 = 1.05 lambda_max at the seed).
# Spectral: sum_k w_k q_k with epigraph variables q_k <= 1 and rows
# ||S_k||_eps + psi rho(T_k) - q_k <= 0 (C more variables). Slack: sum_k ||S_k||_eps + psi s_k tau_hat_k.

"Row (a, b) of pack(L L') as `terms` products, padded with zero-weight products."
function packed_product_terms(a, b, terms)
    pairs = [(packed_index(a, c), packed_index(b, c), 1.0) for c in 1:b]
    while length(pairs) < terms
        push!(pairs, (packed_index(b, b), packed_index(b, b), 0.0))
    end
    return pairs
end

"Named tuple fields a1, b1, m1, ... of the padded products of row (i, j)."
function product_fields(i, j, terms)
    t = packed_product_terms(i, j, terms)
    names = Tuple(Symbol(prefix, k) for k in 1:terms for prefix in (:a, :b, :m))
    values = Tuple(v for k in 1:terms for v in t[k])
    return NamedTuple{names}(values)
end

"""
    StochasticLayout

Problem structure of the stochastic NLP:

* `slots[k]`: control slot of arc k (1..C in arc order; 0: no control);
* `weights[k]`: cost weight of arc k;
* `steps[k]`: propagation time of arc k in the units of its UT parameters;
* `boundary`: fixed node components `(node, i, value)`, one equality row each;
* `defect_rhs`: 7 x n constants of the mean-defect rows means[:, k+1] - ratio .* UT mean
  (origin translations at frame switches; zero otherwise);
* `ratio`: 7 x n end-node units per propagated unit of each arc's UT mean;
* `mean_lower`, `mean_upper`: 7 x (n+1) bounds of the node means;
* `groups`: `(arcs, params)` pairs, the arcs propagated with each set of UT parameters;
* `control_params`: UT parameters of the controlled arcs (executed-control covariance);
* `terminal_transform`: 6 x 6 matrix G of the terminal bound I - G P_N G' >= 0 on the normalized
  position/velocity covariance (diagonal 1 / sqrt(target ratio) for the single-phase cases).
"""
struct StochasticLayout{P<:UTParams}
    slots::Vector{Int}
    weights::Vector{Float64}
    steps::Vector{Float64}
    boundary::Vector{NamedTuple{(:node, :i, :value),Tuple{Int,Int,Float64}}}
    defect_rhs::Matrix{Float64}
    ratio::Matrix{Float64}
    mean_lower::Matrix{Float64}
    mean_upper::Matrix{Float64}
    groups::Vector{Tuple{Vector{Int},P}}
    control_params::P
    terminal_transform::Matrix{Float64}
end

n_arcs(l::StochasticLayout) = length(l.slots)
controlled_arcs(l::StochasticLayout) = findall(>(0), l.slots)

"Layout of a single-phase case: every arc controlled, unit weights, one set of UT parameters."
function single_phase_layout(case::CaseData, o::Options, ref::ReferenceTraj, p::UTParams)
    n = n_arcs(ref)
    target = terminal_mean_target(case, o, ref)
    boundary = [[(node = 1, i = i, value = case.x0[i]) for i in 1:NX];
                [(node = n + 1, i = i, value = target[i]) for i in 1:NP]]
    return StochasticLayout(collect(1:n), single_phase_arc_weights(o, ref), collect(ref.steps), boundary, zeros(NX, n), ones(NX, n),
        fill(-Inf, NX, n + 1), fill(Inf, NX, n + 1), [(collect(1:n), p)], p, Matrix(Diagonal(terminal_inverse_std(o))))
end

"""
MadNLP's initial point of a variable in [l, u] (`initialize_variables!`; bound_fac = 1e-2, MadNLP's default,
and `push` the solve's bound_push).
"""
function madnlp_start(x, l, u; push = 1e-2, fac = 1e-2)
    isfinite(l) && isfinite(u) && return min(u - min(push * max(1, abs(u)), fac * (u - l)), max(l + min(push * max(1, abs(l)), fac * (u - l)), x))
    isfinite(l) && return max(l + push * max(1, abs(l)), x)
    isfinite(u) && return min(u - push * max(1, abs(u)), x)
    return x
end

"Packed start factor of the arc starting at `node` (the fixed initial factor on the first node)."
start_factor(scaling::CovarianceScaling, node) = node == 1 ? scaling.initial_factor : scaling.factors[:, node-1]

"Control covariance T_k of the arcs `arcs` at the model's start point, as `control_covariance` evaluates it."
function start_control_covariances(seed::InitialGuess, scaling::CovarianceScaling, p::UTParams, arcs)
    return map(arcs) do a
        P6 = position_velocity_covariance(static_node_scale(scaling, a), SVector{21}(start_factor(scaling, a)[1:21]))
        Matrix(control_covariance(p, SVector{3}(seed.feedforward[:, a]), SMatrix{3,6}(seed.gains[:, :, a]), P6))
    end
end

"""
Scale s = sqrt(lambda_max(T)) and the scaled start of the slack chance variables: tau^2 =
lambda_max + 4 floor^2 (the smallest G pivot starts near 2 floor) and G = chol(tau^2 I - T).
lambda_max comes from the LAPACK symmetric eigensolver, not the smoothed formula.
"""
function slack_chance_start(T::AbstractMatrix, floor)
    lambda_max = eigmax(Symmetric(T))
    s = sqrt(max(lambda_max, 1e-16))
    tau = sqrt(lambda_max + 4floor^2)
    G = cholesky(Symmetric(tau^2 * I - T)).L
    return s, tau / s, pack_lower(Matrix(G)) ./ s
end

"""
UT oracle layout of the arcs `arcs` (one group): local input indices of the add_eval
variable list (means, ff, gains, chol, ...), fixed inputs and output rows of the
(mean_defects, cov_defects) blocks.
"""
function group_ut_layout(l::StochasticLayout, arcs, offsets, scaling::CovarianceScaling, seed::InitialGuess;
        directions = nothing, inputs = NTHETA)
    n, m = n_arcs(l), length(arcs)
    theta_index = zeros(Int, inputs, m)
    theta_fixed = zeros(inputs, m)
    rows = zeros(Int, NUT_ROWS, m)
    for (a, k) in enumerate(arcs)
        theta_index[1:7, a] = offsets.means .+ 7(k-1) .+ (1:7)
        slot = l.slots[k]
        if slot > 0
            theta_index[8:10, a] = offsets.ff .+ 3(slot-1) .+ (1:3)
            directions === nothing || (theta_fixed[8:10, a] = directions[:, slot])   # polar seed direction
            inputs > NTHETA && (theta_index[57, a] = offsets.throttle + slot)          # throttle Gamma_k
            if inputs > NTHETA + 1
                # lifted commands (eta, e): the gains reach the oracle only through the model's lifting rows
                theta_index[LIFT_BASE+1:inputs, a] = offsets.lift .+ 4NLIFT * (slot-1) .+ (1:4NLIFT)
            else
                theta_index[11:28, a] = offsets.gains .+ 18(slot-1) .+ (1:18)
            end
        end
        if k == 1
            theta_fixed[29:56, a] = scaling.initial_factor
        else
            theta_index[29:56, a] = offsets.chol .+ 28(k-2) .+ (1:28)
        end
        rows[1:NX, a] = 7(k-1) .+ (1:NX)
        rows[NX+1:end, a] = 7n + 28(k-1) .+ (1:NL)
    end
    arc_std = scaling.full ? packed_node_scales(scaling, arcs) : scaling.std[:, arcs]
    inv_next = scaling.full ? packed_inverse_node_scales(scaling, arcs .+ 1) : scaling.inv_std[:, arcs .+ 1]
    return (; theta_index, theta_fixed, rows, ratio = l.ratio[:, arcs], arc_std, inv_next,
        gain_scale = seed.gain_scale[arcs], steps = l.steps[arcs], full = Val(scaling.full))
end

"Node scale of node k as a static vector of standard deviations (diagonal) or lower matrix (full)."
static_node_scale(scaling::CovarianceScaling, k) =
    scaling.full ? SMatrix{7,7}(scaling.scale[:, :, k]) : SVector{7}(scaling.std[:, k])

"Named tuple fields of `terms` products (gain column g, factor entry r, coefficient c), padded with zero terms."
function gain_factor_fields(terms, count)
    terms = [terms; fill((1, 1, 0.0), count - length(terms))]
    names = Tuple(Symbol(prefix, j) for j in 1:count for prefix in (:g, :r, :c))
    return NamedTuple{names}(Tuple(v for t in terms for v in t))
end

"""
State-covariance trace and feedback-covariance cost terms with full lower node scales,
P_k = (S_k L_k)(S_k L_k)' on the arc start node: tr(P6) sums the squares of
(S L)_ic = sum_{m=c..i} S_im L_mc (up to 6 factor entries), and tr(K P6 K') the squares of
(K S L)_{q,col} = sum_{row >= m >= col} K_{q,row} S_{row,m} L_{m,col} (up to 21 products).
"""
function add_full_scale_tracking(c, l::StochasticLayout, controlled, scaling::CovarianceScaling, seed::InitialGuess,
        p::UTParams, cA, gains, chol)
    n = n_arcs(l)
    trace_data = [begin
        S = scaling.scale[:, :, a]
        terms = [(m, packed_index(m, col), S[i, m]) for m in col:i]
        merge((k = a - 1, w = p.inv_sigma_sq * l.weights[a]), gain_factor_fields(terms, 6))
    end for a in 2:n for i in 1:NP for col in 1:i]
    @add_obj(c, t.w * (t.c1 * chol[t.r1, t.k] + t.c2 * chol[t.r2, t.k] + t.c3 * chol[t.r3, t.k] +
        t.c4 * chol[t.r4, t.k] + t.c5 * chol[t.r5, t.k] + t.c6 * chol[t.r6, t.k])^2 for t in trace_data)
    kp_data = [begin
        S = scaling.scale[:, :, a]
        terms = [(row, packed_index(m, col), S[row, m]) for row in col:NP for m in col:row]
        merge((a = l.slots[a], k = a - 1, q = q, w = cA * p.ct * p.d * seed.gain_scale[a]^2 * l.weights[a]),
            gain_factor_fields(terms, 21))
    end for a in filter(>=(2), controlled) for q in 1:NU for col in 1:NP]
    isempty(kp_data) || @add_obj(c, t.w * (
        t.c1 * gains[t.q, t.g1, t.a] * chol[t.r1, t.k] + t.c2 * gains[t.q, t.g2, t.a] * chol[t.r2, t.k] +
        t.c3 * gains[t.q, t.g3, t.a] * chol[t.r3, t.k] + t.c4 * gains[t.q, t.g4, t.a] * chol[t.r4, t.k] +
        t.c5 * gains[t.q, t.g5, t.a] * chol[t.r5, t.k] + t.c6 * gains[t.q, t.g6, t.a] * chol[t.r6, t.k] +
        t.c7 * gains[t.q, t.g7, t.a] * chol[t.r7, t.k] + t.c8 * gains[t.q, t.g8, t.a] * chol[t.r8, t.k] +
        t.c9 * gains[t.q, t.g9, t.a] * chol[t.r9, t.k] + t.c10 * gains[t.q, t.g10, t.a] * chol[t.r10, t.k] +
        t.c11 * gains[t.q, t.g11, t.a] * chol[t.r11, t.k] + t.c12 * gains[t.q, t.g12, t.a] * chol[t.r12, t.k] +
        t.c13 * gains[t.q, t.g13, t.a] * chol[t.r13, t.k] + t.c14 * gains[t.q, t.g14, t.a] * chol[t.r14, t.k] +
        t.c15 * gains[t.q, t.g15, t.a] * chol[t.r15, t.k] + t.c16 * gains[t.q, t.g16, t.a] * chol[t.r16, t.k] +
        t.c17 * gains[t.q, t.g17, t.a] * chol[t.r17, t.k] + t.c18 * gains[t.q, t.g18, t.a] * chol[t.r18, t.k] +
        t.c19 * gains[t.q, t.g19, t.a] * chol[t.r19, t.k] + t.c20 * gains[t.q, t.g20, t.a] * chol[t.r20, t.k] +
        t.c21 * gains[t.q, t.g21, t.a] * chol[t.r21, t.k])^2 for t in kp_data)
    return c
end

"""
    build_stochastic_nlp(layout, o, seed, scaling, backend; extra_constraints) -> blocks

The ExaModels model of `layout` and its variable blocks. `extra_constraints(core, blocks, nvar)`
may append case-specific rows after the chance rows and returns the core. `uts` holds the UT
oracles (index data and workspace; used by the tools that call them directly).
"""
function build_stochastic_nlp(l::StochasticLayout, o::Options, seed::InitialGuess, scaling::CovarianceScaling,
        backend; extra_constraints = nothing)
    n = n_arcs(l)
    controlled = controlled_arcs(l)
    nc = length(controlled)
    l.slots[controlled] == 1:nc || error("control slots must number the controlled arcs in order")
    p = l.control_params
    c = ExaCore(Float64; backend)
    lvar_chol = fill(-Inf, NL, n)
    lvar_chol[collect(LTRI_DIAGONAL), :] .= 0.0
    lvar_margin = fill(-Inf, NM)
    lvar_margin[collect(MTRI_DIAGONAL)] .= o.terminal_margin_floor
    gs = seed.gain_scale[controlled]
    dispersion = o.objective == :terminal_dispersion
    margin_start = seed.terminal_margin
    if dispersion
        GPG = l.terminal_transform * seed.covariances[1:NP, 1:NP, end] * l.terminal_transform'
        level_start = 1.05 * eigmax(Symmetric(GPG))
        margin_start = pack_lower(Matrix(cholesky(Symmetric(level_start * I - GPG)).L))
    end
    @add_var(c, means, 1:NX, 1:n+1; start = seed.means, lvar = l.mean_lower, uvar = l.mean_upper)
    mode = o.control_parameterization
    polar, throttled = mode == :polar, mode == :throttle
    directions, gamma0, gamma_floor = nothing, nothing, 0.0
    if mode == :cartesian
        @add_var(c, ff, 1:NU, 1:nc; start = seed.feedforward[:, controlled])
    else
        # Gamma = ||S_seed|| with the seed direction reproduces the seed feedforward. Gamma >= 0 is relaxed by
        # MadNLP's bound_relax_factor (0 on the CPU, condensed_relaxation on the GPU): this lower bound puts
        # the relaxed bound at exactly 0, so Gamma > 0 at every iterate and ||Gamma v|| = Gamma ||v||.
        S0 = seed.feedforward[:, controlled]
        gamma0 = [norm(S0[:, k]) for k in 1:nc]
        directions = reduce(hcat, [gamma0[k] > 0 ? S0[:, k] / gamma0[k] : [1.0, 0.0, 0.0] for k in 1:nc])
        gamma_floor = o.execution_mode == :gpu ? o.condensed_relaxation : 0.0
        if polar
            @add_var(c, ff, 1:NU, 1:nc; start = vcat(gamma0', zeros(2, nc)),
                lvar = vcat(fill(gamma_floor, 1, nc), fill(-Inf, 2, nc)))
        else
            @add_var(c, ff, 1:NU, 1:nc; start = directions)   # direction d_k (rows ||d_k||^2 = 1)
        end
    end
    @add_var(c, gains, 1:NU, 1:NP, 1:nc; start = seed.gains[:, :, controlled] ./ reshape(gs, 1, 1, :),
        lvar = fill(-o.gain_bound, NU, NP, nc), uvar = fill(o.gain_bound, NU, NP, nc))
    lifting = throttled && lifted(p)
    # With lifted commands the factors and the throttle start at MadNLP's initial point (`madnlp_start`; lifting_start
    # = :pushed): its push would move Gamma and the factor diagonals, i.e. the commands of the unlifted sigma points,
    # but not eta e. lifting_start = :seed keeps the seed values.
    pushed = lifting && o.lifting_start == :pushed
    chol_start = pushed ? madnlp_start.(scaling.factors, lvar_chol, Inf; push = o.bound_push) : scaling.factors
    @add_var(c, chol, 1:NL, 1:n; start = chol_start, lvar = lvar_chol)
    throttle = nothing
    if throttled
        # right after the factors: the UT oracle's variable list (means, ff, gains, chol, throttle) stays contiguous
        gamma_start = pushed ? madnlp_start.(gamma0, gamma_floor, 1.0; push = o.bound_push) : gamma0
        @add_var(c, throttle, 1:nc; start = gamma_start, lvar = fill(gamma_floor, nc), uvar = ones(nc))
    end
    lift = nothing
    if lifting
        # right after the throttle (the oracle's variable list stays contiguous), started at the commands c_o of
        # the model's start point as the unlifted oracle forms them: eta_o = ||c_o||, e_o = c_o / eta_o
        start_x = vcat(vec(seed.means), vec(directions), vec(seed.gains[:, :, controlled] ./ reshape(gs, 1, 1, :)),
            vec(chol_start), gamma_start)
        start_offsets = (; means = means.offset, ff = ff.offset, gains = gains.offset, chol = chol.offset,
            throttle = throttle.offset)
        lift_start = zeros(4, NLIFT, nc)
        for (arcs, params) in l.groups
            lifted(params) || continue
            commands = lifted_commands(group_ut_layout(l, arcs, start_offsets, scaling, seed; inputs = NTHETA + 1),
                params, start_x)
            for (a, k) in enumerate(arcs), m in 1:NLIFT
                slot = l.slots[k]
                slot > 0 || continue
                eta = norm(commands[:, m, a])
                lift_start[:, m, slot] = eta > 0 ? [eta; commands[:, m, a] / eta] : [0.0, 1.0, 0.0, 0.0]
            end
        end
        # eta is free: (eta, e) and (-eta, -e) give the same command, and the mass flow, the Gates covariance and the
        # executed points depend on the vector c only. A bound eta >= 0 would be pushed to 1e-2 at the start (MadNLP's
        # bound_push) and held there by the barrier, 20 times the coast-arc feedback.
        @add_var(c, lift, 1:4, 1:NLIFT, 1:nc; start = lift_start)
    end
    terminal_bound = o.terminal_covariance == :constraint
    margin = nothing
    if terminal_bound
        @add_var(c, margin, 1:NM; start = margin_start, lvar = lvar_margin)
    end
    norm_epigraph = o.norm_epigraph
    thrust = nothing
    if norm_epigraph
        @add_var(c, thrust, 1:nc; start = [norm(seed.feedforward[:, a]) + 1e-3 for a in controlled], lvar = zeros(nc))
    end
    slack = o.chance_formulation == :slack
    # polar and throttle controls: the epigraph rows of `chance_slack_oracle` (any layout, full node scales, the
    # throttle input); the cartesian controls keep the algebraic rows of `add_slack_chance`
    slack_oracle = slack && mode != :cartesian
    if slack_oracle
        starts = [slack_oracle_start(T, o.chance_slack_floor) for T in start_control_covariances(seed, scaling, p, controlled)]
        slack_scale = [first(s) for s in starts]
        lvar_slack = fill(-Inf, NSLACK, nc)
        for k in 1:nc, r in 1:NSLACK
            SLACK_ROWS[r] == SLACK_COLS[r] && (lvar_slack[r, k] = o.chance_slack_floor / slack_scale[k])
        end
        @add_var(c, tau, 1:nc; start = [s[2] for s in starts], lvar = zeros(nc))
        @add_var(c, slack_factor, 1:NSLACK, 1:nc; start = reduce(hcat, [s[3] for s in starts]), lvar = lvar_slack)
    elseif slack
        controlled == 1:n || error("the slack chance formulation needs every arc controlled")
        scaling.full && error("the slack chance formulation needs diagonal node scales")
        starts = [slack_chance_start(T, o.chance_slack_floor) for T in start_control_covariances(seed, scaling, p, 1:n)]
        slack_scale = [first(s) for s in starts]
        lvar_slack = fill(-Inf, NU * (NU + 1) ÷ 2, n)
        for a in 1:n, r in 1:NU
            lvar_slack[packed_index(r, r), a] = o.chance_slack_floor / slack_scale[a]
        end
        @add_var(c, tau, 1:n; start = [s[2] for s in starts], lvar = zeros(n))
        @add_var(c, slack_factor, 1:NU*(NU+1)÷2, 1:n; start = reduce(hcat, [s[3] for s in starts]), lvar = lvar_slack)
    end

    # ------------------------------------------------------------ objective
    # A feasibility solve has no objective terms (and no objective constant).
    quantile = o.objective == :quantile && !o.feasibility_only
    chance_sum = o.objective == :chance_sum && !o.feasibility_only
    epigraph = quantile && !slack
    constant = 0.0
    level = nothing
    if dispersion
        @add_var(c, level, 1:1; start = [level_start], lvar = zeros(1))
        @add_obj(c, level[i] for i in 1:1)
    elseif quantile && slack
        quantile_data = [(k = a, w = p.psi * slack_scale[a], e2 = p.eps^2) for a in 1:n]
        @add_obj(c, sqrt(ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2 + t.e2) + t.w * tau[t.k] for t in quantile_data)
    elseif chance_sum && slack_oracle
        # sum_k w_k (Gamma_k + psi s_k tau_hat_k): the chance left-hand sides with the epigraph tau_hat
        cost_data = [(k = k, w = l.weights[a], ps = p.psi * slack_scale[k]) for (k, a) in enumerate(controlled)]
        if throttled
            @add_obj(c, t.w * (throttle[t.k] + t.ps * tau[t.k]) for t in cost_data)
        else
            @add_obj(c, t.w * (ff[1, t.k] + t.ps * tau[t.k]) for t in cost_data)
        end
    elseif chance_sum
        # no algebraic term: sum_k w_k (chance row k) is added after the ExaModel is built
    elseif epigraph
        start = [chance_value(p, static_node_scale(scaling, a), seed.gain_scale[a], SVector{3}(seed.feedforward[:, a]),
            SVector{18}(vec(seed.gains[:, :, a] ./ seed.gain_scale[a])), SVector{21}(start_factor(scaling, a)[1:21]), Val(true))
            for a in controlled]
        @add_var(c, q, 1:nc; start = start, uvar = ones(nc))
        @add_obj(c, t.w * q[t.k] for t in [(k = l.slots[a], w = l.weights[a]) for a in controlled])
    elseif !o.feasibility_only
        norm_data = [(k = l.slots[a], w = l.weights[a], e2 = o.control_norm_eps^2) for a in controlled]
        if norm_epigraph
            @add_obj(c, t.w * thrust[t.k] for t in norm_data)
        elseif polar
            @add_obj(c, t.w * ff[1, t.k] for t in norm_data)   # ||S_k|| = Gamma_k
        elseif throttled
            @add_obj(c, t.w * throttle[t.k] for t in norm_data)
        else
            @add_obj(c, t.w * sqrt(ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2 + t.e2) for t in norm_data)
        end
        cA = o.control_covariance_weight * (1 + p.var_mag + 2p.var_point)
        cf = o.control_covariance_weight * (p.var_mag + 2p.var_point)
        F1 = scaled_factor(scaling, 1, unpack_lower(scaling.initial_factor, NX))
        constant += l.weights[1] * tr((F1 * F1')[1:NP, 1:NP]) * p.inv_sigma_sq
        if scaling.full
            c = add_full_scale_tracking(c, l, controlled, scaling, seed, p, cA, gains, chol)
        else
            trace_data = [(k = a - 1, r = r, w = scaling.std[LTRI_ROWS[r], a]^2 * p.inv_sigma_sq * l.weights[a])
                          for a in 2:n for r in 1:NM]
            @add_obj(c, t.w * chol[t.r, t.k]^2 for t in trace_data)
            # tr(K P6 K') = sum_{q, col} (sum_{row >= col} K_{q,row} std_row L_{row,col})^2
            kp_data = [begin
                weights = [row >= col ? scaling.std[row, a] : 0.0 for row in 1:NP]
                index(row) = packed_index(max(row, col), col)
                (a = l.slots[a], k = a - 1, q = q, w = cA * p.ct * p.d * seed.gain_scale[a]^2 * l.weights[a],
                 c1 = weights[1], r1 = index(1), c2 = weights[2], r2 = index(2), c3 = weights[3], r3 = index(3),
                 c4 = weights[4], r4 = index(4), c5 = weights[5], r5 = index(5), c6 = weights[6], r6 = index(6))
            end for a in filter(>=(2), controlled) for q in 1:NU for col in 1:NP]
            isempty(kp_data) || @add_obj(c, t.w * (t.c1 * gains[t.q, 1, t.a] * chol[t.r1, t.k] +
                t.c2 * gains[t.q, 2, t.a] * chol[t.r2, t.k] + t.c3 * gains[t.q, 3, t.a] * chol[t.r3, t.k] +
                t.c4 * gains[t.q, 4, t.a] * chol[t.r4, t.k] + t.c5 * gains[t.q, 5, t.a] * chol[t.r5, t.k] +
                t.c6 * gains[t.q, 6, t.a] * chol[t.r6, t.k])^2 for t in kp_data)
        end
        # tr(K P6 K') on arcs starting at the fixed initial node: (K F1)_{q,col}, F1 the scaled initial factor
        kp0_data = [begin
            weights = [row >= col ? F1[row, col] : 0.0 for row in 1:NP]
            (a = l.slots[1], q = q, w = cA * p.ct * p.d * seed.gain_scale[1]^2 * l.weights[1], c1 = weights[1],
             c2 = weights[2], c3 = weights[3], c4 = weights[4], c5 = weights[5], c6 = weights[6])
        end for _ in filter(==(1), controlled) for q in 1:NU for col in 1:NP]
        isempty(kp0_data) || @add_obj(c, t.w * (t.c1 * gains[t.q, 1, t.a] + t.c2 * gains[t.q, 2, t.a] +
            t.c3 * gains[t.q, 3, t.a] + t.c4 * gains[t.q, 4, t.a] + t.c5 * gains[t.q, 5, t.a] +
            t.c6 * gains[t.q, 6, t.a])^2 for t in kp0_data)
        if isdiag(p.nav_cov)
            nav_data = [(a = l.slots[a], q = q, b = b,
                         w = cA * p.ct * seed.gain_scale[a]^2 * (p.d * p.nav_cov[b, b]) * l.weights[a])
                        for a in controlled for b in 1:NP for q in 1:NU]
            @add_obj(c, t.w * gains[t.q, t.b, t.a]^2 for t in nav_data)
        else
            # d tr(K R_nav K') = sum_{q, col} (sum_row K_{q,row} F_{row,col})^2 with F F' = d R_nav
            F = p.nav_factor
            nav_data = [(a = l.slots[a], q = q, w = cA * p.ct * seed.gain_scale[a]^2 * l.weights[a],
                         c1 = F[1, col], c2 = F[2, col], c3 = F[3, col], c4 = F[4, col], c5 = F[5, col], c6 = F[6, col])
                        for a in controlled for q in 1:NU for col in 1:NP]
            @add_obj(c, t.w * (t.c1 * gains[t.q, 1, t.a] + t.c2 * gains[t.q, 2, t.a] + t.c3 * gains[t.q, 3, t.a] +
                               t.c4 * gains[t.q, 4, t.a] + t.c5 * gains[t.q, 5, t.a] + t.c6 * gains[t.q, 6, t.a])^2
                for t in nav_data)
        end
        if polar
            @add_obj(c, t.w * ff[1, t.k]^2 for t in [(k = l.slots[a], w = cf * l.weights[a]) for a in controlled])
        elseif throttled
            @add_obj(c, t.w * throttle[t.k]^2 * (ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2)
                for t in [(k = l.slots[a], w = cf * l.weights[a]) for a in controlled])
        else
            @add_obj(c, t.w * (ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2)
                for t in [(k = l.slots[a], w = cf * l.weights[a]) for a in controlled])
        end
        if !terminal_bound
            # w_N tr(G P_N G')/c^2 = w_N/c^2 sum_{i, col} (sum_k M_ik L_N[k, col])^2 with M = G S_N
            M = l.terminal_transform * node_scale(scaling, n + 1)[1:NP, 1:NP]
            wN = o.terminal_covariance_weight / o.bryson_sigma_factor^2
            terminal_cost = [begin
                row = [k >= col ? M[i, k] : 0.0 for k in 1:NP]
                index = [packed_index(max(k, col), col) for k in 1:NP]
                (w = wN, c1 = row[1], r1 = index[1], c2 = row[2], r2 = index[2], c3 = row[3], r3 = index[3],
                 c4 = row[4], r4 = index[4], c5 = row[5], r5 = index[5], c6 = row[6], r6 = index[6])
            end for i in 1:NP for col in 1:NP if any(!iszero, M[i, col:NP])]
            @add_obj(c, t.w * (t.c1 * chol[t.r1, n] + t.c2 * chol[t.r2, n] + t.c3 * chol[t.r3, n] +
                t.c4 * chol[t.r4, n] + t.c5 * chol[t.r5, n] + t.c6 * chol[t.r6, n])^2 for t in terminal_cost)
        end
        constant += sum(o.control_covariance_weight * 3p.jitter * l.weights[a] for a in controlled; init = 0.0)
    end

    # ---------------------------------------------------------- constraints
    @add_con(c, means[t.i, t.node] - t.value for t in l.boundary)
    # Native parts of the defect rows; the UT oracles below add the predicted moments.
    # The origin translations of switch arcs enter the rows, not their bounds: a relaxed equality (GPU,
    # bound_relax_factor) then keeps a band of 1e-8, not 1e-8 times the translation.
    @add_con(c, mean_defects, means[t.i, t.k+1] - t.rhs for t in [(i = i, k = k, rhs = l.defect_rhs[i, k]) for k in 1:n for i in 1:NX])
    factor_defects = o.covariance_defect != :product
    cov_data = [merge((k = a,), product_fields(LTRI_ROWS[r], LTRI_COLS[r], NX)) for a in 1:n for r in 1:NL]
    if factor_defects
        @add_con(c, cov_defects, chol[t.r, t.k] for t in [(k = a, r = r) for a in 1:n for r in 1:NL])
    else
        @add_con(c, cov_defects, t.m1 * chol[t.a1, t.k] * chol[t.b1, t.k] + t.m2 * chol[t.a2, t.k] * chol[t.b2, t.k] +
            t.m3 * chol[t.a3, t.k] * chol[t.b3, t.k] + t.m4 * chol[t.a4, t.k] * chol[t.b4, t.k] +
            t.m5 * chol[t.a5, t.k] * chol[t.b5, t.k] + t.m6 * chol[t.a6, t.k] * chol[t.b6, t.k] +
            t.m7 * chol[t.a7, t.k] * chol[t.b7, t.k] for t in cov_data)
    end
    # Terminal covariance bound G P_N G' <= I (matrix inequality):  I - G P_N G' - M M' = 0
    # (t I in place of I with objective = :terminal_dispersion).
    G = l.terminal_transform
    std_N = scaling.std[:, n+1]
    if !terminal_bound
    elseif isdiag(G) && !scaling.full
        inverse_target = diag(G)
        terminal_data = [begin
            i, j = LTRI_ROWS[r], LTRI_COLS[r]
            merge((delta = Float64(i == j && !dispersion), w = std_N[i] * std_N[j] * inverse_target[i] * inverse_target[j]),
                product_fields(i, j, NP))
        end for r in 1:NM]
        @add_con(c, terminal_rows, t.delta -
            t.w * (t.m1 * chol[t.a1, n] * chol[t.b1, n] + t.m2 * chol[t.a2, n] * chol[t.b2, n] +
                   t.m3 * chol[t.a3, n] * chol[t.b3, n] + t.m4 * chol[t.a4, n] * chol[t.b4, n] +
                   t.m5 * chol[t.a5, n] * chol[t.b5, n] + t.m6 * chol[t.a6, n] * chol[t.b6, n]) -
            (t.m1 * margin[t.a1] * margin[t.b1] + t.m2 * margin[t.a2] * margin[t.b2] +
             t.m3 * margin[t.a3] * margin[t.b3] + t.m4 * margin[t.a4] * margin[t.b4] +
             t.m5 * margin[t.a5] * margin[t.b5] + t.m6 * margin[t.a6] * margin[t.b6]) for t in terminal_data)
    else
        # (M L_N)(M L_N)' with M = G S_N (S_N the node scale): row (i, j) collects M_ik M_jm L_kc L_mc.
        M = G * node_scale(scaling, n + 1)[1:NP, 1:NP]
        terminal_data = [merge((delta = Float64(LTRI_ROWS[r] == LTRI_COLS[r] && !dispersion),),
                               product_fields(LTRI_ROWS[r], LTRI_COLS[r], NP)) for r in 1:NM]
        @add_con(c, terminal_rows, t.delta -
            (t.m1 * margin[t.a1] * margin[t.b1] + t.m2 * margin[t.a2] * margin[t.b2] +
             t.m3 * margin[t.a3] * margin[t.b3] + t.m4 * margin[t.a4] * margin[t.b4] +
             t.m5 * margin[t.a5] * margin[t.b5] + t.m6 * margin[t.a6] * margin[t.b6]) for t in terminal_data)
        terms = [(id = r, p1 = packed_index(k, col), p2 = packed_index(m, col),
                  w = -M[LTRI_ROWS[r], k] * M[LTRI_COLS[r], m])
                 for r in 1:NM for col in 1:NP for k in col:NP for m in col:NP
                 if M[LTRI_ROWS[r], k] != 0 && M[LTRI_COLS[r], m] != 0]
        @add_con!(c, terminal_rows, t.id => t.w * chol[t.p1, n] * chol[t.p2, n] for t in terms)
    end
    if dispersion && terminal_bound
        @add_con!(c, terminal_rows, t.id => level[t.v] for t in [(id = r, v = 1) for r in 1:NM if LTRI_ROWS[r] == LTRI_COLS[r]])
    end

    # --------------------------------------------------------------- oracles
    # add_eval's variable list starts at the first variable, so its local indices are global.
    offsets = (; means = means.offset, ff = ff.offset, gains = gains.offset, chol = chol.offset,
        throttle = throttled ? throttle.offset : nothing, lift = lifting ? lift.offset : nothing)
    offsets.means == 0 && offsets.ff == NX * (n + 1) || error("unexpected variable layout")
    throttled && offsets.throttle != chol.offset + NL * n && error("the throttle must follow the factors")
    lifting && offsets.lift != offsets.throttle + nc && error("the lifted commands must follow the throttle")
    sort(reduce(vcat, first.(l.groups))) == 1:n || error("the UT groups must cover every arc once")
    oracle_directions = polar ? directions : nothing
    oracle_variables = lifting ? (means, ff, gains, chol, throttle, lift) :
        throttled ? (means, ff, gains, chol, throttle) : (means, ff, gains, chol)
    uts = UTOracle[]
    for (arcs, params) in l.groups
        ut = ut_oracle(group_ut_layout(l, arcs, offsets, scaling, seed; directions = oracle_directions,
            inputs = ntheta(params)), params, backend)
        fu = factor_defects ? ut_factor_oracle(ut; qr = o.covariance_defect == :qr) : nothing
        c, _ = ExaModels.add_eval(c, (mean_defects, cov_defects), oracle_variables,
            factor_defects ? (out, x) -> ut_factor_values!(out, x, fu) : (out, x) -> ut_values!(out, x, ut);
            jac! = factor_defects ? (vals, x) -> ut_factor_jacobian!(vals, x, fu) : (vals, x) -> ut_jacobian!(vals, x, ut),
            hess! = factor_defects ? (vals, x, y) -> ut_factor_hessian!(vals, x, y, fu) :
                (vals, x, y) -> ut_hessian!(vals, x, y, ut),
            jac_structure! = (r, cc) -> (append!(r, ut.structure.jac_rows); append!(cc, ut.structure.jac_cols)),
            hess_structure! = (r, cc) -> (append!(r, ut.structure.hess_rows); append!(cc, ut.structure.hess_cols)))
        push!(uts, ut)
    end
    nvar = NX * (n + 1) + (NU + NU * NP) * nc + NL * n + NM * terminal_bound + nc * norm_epigraph + (slack_oracle ? 7nc : slack ? 7n : 0) +
        (epigraph ? nc : 0) + dispersion + nc * throttled + 4NLIFT * nc * lifting
    if slack_oracle
        chance_data = [(k = k, w = p.psi * slack_scale[k]) for k in 1:nc]
        if throttled
            @add_con(c, throttle[t.k] + t.w * tau[t.k] for t in chance_data; lcon = -Inf, ucon = 1.0)
        else
            @add_con(c, ff[1, t.k] + t.w * tau[t.k] for t in chance_data; lcon = -Inf, ucon = 1.0)
        end
        layout = chance_layout(l.slots[controlled], controlled, offsets, scaling, gs; directions = oracle_directions,
            throttle = offsets.throttle)
        c = ExaModels.constraint(c, chance_slack_oracle(layout, nvar, p, backend, tau.offset, slack_factor.offset,
            1.0 ./ slack_scale .^ 2))
    elseif slack
        c = add_slack_chance(c, n, scaling, seed, p, ff, gains, chol, tau, slack_factor, slack_scale)
    else
        layout = chance_layout(l.slots[controlled], controlled, offsets, scaling, gs; directions = oracle_directions,
            throttle = offsets.throttle)
        chance_cost = chance_sum ? chance_oracle(layout, nvar, p, backend) : nothing   # not with slack rows
        c = ExaModels.constraint(c, chance_oracle(layout, nvar, p, backend; epigraph = epigraph ? q.offset : nothing,
            norm_slack = norm_epigraph ? thrust.offset : nothing))
    end
    if norm_epigraph
        @add_con(c, thrust[t.k]^2 - ff[1, t.k]^2 - ff[2, t.k]^2 - ff[3, t.k]^2 for t in [(k = k,) for k in 1:nc];
            lcon = zeros(nc), ucon = fill(Inf, nc))
    end
    if throttled
        @add_con(c, unit_direction, ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2 - 1 for t in [(k = k,) for k in 1:nc];
            lcon = fill(o.direction_constraint == :ball ? -Inf : 0.0, nc), ucon = zeros(nc))
    end
    lifting && (c = add_command_lifting(c, l, controlled, scaling, seed, p, ff, gains, chol, throttle, lift))
    blocks = (; means, ff, gains, chol, margin, throttle, lift)
    extra_constraints === nothing || (c = extra_constraints(c, blocks, nvar))
    model = ExaModel(c)
    model.meta.nvar == nvar || error("variable count $(model.meta.nvar) differs from the expected $nvar")
    chance_sum && !slack && (model = ChanceObjectiveModel(model, chance_cost, l.weights[controlled], backend))
    ncon = length(l.boundary) + (NX + NL) * n + NM * terminal_bound + nc + nc * norm_epigraph + (slack_oracle ? 6nc : slack ? 6n : 0) +
        nc * throttled + 4NLIFT * nc * lifting
    extra_constraints === nothing && model.meta.ncon != ncon &&
        error("constraint count $(model.meta.ncon) differs from the expected $ncon")
    constant = o.feasibility_only || quantile || dispersion || chance_sum ? 0.0 : constant
    chance_slack = slack ? (; tau, factor = slack_factor, scale = slack_scale) : nothing
    return (; model, means, ff, gains, chol, margin, level, thrust, throttle, lift, chance_slack, objective_constant = constant,
        uts, ut = first(uts), controlled, polar_directions = oracle_directions)
end

"""
Lifting rows of the sigma-point commands (`sigma_command_lifting`): for every controlled arc a (slot k) and lifted
outer point m (feedback column j, sign sigma; `lift_index`)

    Gamma_k d_k + sigma gs_a K_hat_k m_o - eta_m e_m = 0   (3 rows),   ||e_m||^2 - 1 = 0,

with m_o the measurement of column j as the UT oracle forms it: the spread sqrt(d) (S_a L_a)[1:6, j] of a state
column (S_a the start node scale, L_a the start factor: data on the first node) or a navigation factor column.
"""
function add_command_lifting(c, l::StochasticLayout, controlled, scaling::CovarianceScaling, seed::InitialGuess,
        p::UTParams, ff, gains, chol, throttle, lift)
    rows = [(k = l.slots[a], i = i, e = 1 + i, m = m) for a in controlled for m in 1:NLIFT for i in 1:NU]
    row_of = Dict((r.k, r.m, r.i) => id for (id, r) in enumerate(rows))
    @add_con(c, lifting_rows, throttle[t.k] * ff[t.i, t.k] - lift[1, t.m, t.k] * lift[t.e, t.m, t.k] for t in rows)
    products, linear = [], []
    for a in controlled, o in 1:2p.d
        m = lift_index(p, o)
        m == 0 && continue
        k, j, sigma = l.slots[a], (o - 1) % p.d + 1, o <= p.d ? 1.0 : -1.0
        S = node_scale(scaling, a)
        for i in 1:NU, col in 1:NP
            id, w = row_of[(k, m, i)], sigma * seed.gain_scale[a]
            if j > NX
                coefficient = w * p.nav_factor[col, j - NX]
                coefficient == 0 || push!(linear, (id = id, i = i, col = col, k = k, w = coefficient))
                continue
            end
            for r in j:col       # m_o[col] = sqrt(d) sum_{r = j..col} S[col, r] L[r, j]
                coefficient = w * p.sqrt_d * S[col, r]
                coefficient == 0 && continue
                if a == 1
                    push!(linear, (id = id, i = i, col = col, k = k,
                        w = coefficient * scaling.initial_factor[packed_index(r, j)]))
                else
                    push!(products, (id = id, i = i, col = col, k = k, r = packed_index(r, j), node = a - 1, w = coefficient))
                end
            end
        end
    end
    products, linear = identity.(products), identity.(linear)
    isempty(products) || @add_con!(c, lifting_rows, t.id => t.w * gains[t.i, t.col, t.k] * chol[t.r, t.node] for t in products)
    isempty(linear) || @add_con!(c, lifting_rows, t.id => t.w * gains[t.i, t.col, t.k] for t in linear)
    nc = length(controlled)
    @add_con(c, lift[2, t.m, t.k]^2 + lift[3, t.m, t.k]^2 + lift[4, t.m, t.k]^2 - 1 for t in [(k = k, m = m) for k in 1:nc for m in 1:NLIFT])
    return c
end

"The stochastic NLP of a single-phase case."
build_stochastic_model(case::CaseData, o::Options, ref::ReferenceTraj, seed::InitialGuess,
        scaling::CovarianceScaling, p::UTParams, backend) =
    build_stochastic_nlp(single_phase_layout(case, o, ref, p), o, seed, scaling, backend)

"""
Slack thrust chance constraint (`chance_formulation = :slack`): the rows
||S_k||_eps + psi s_k tau_hat_k <= 1, then for every arc the packed lower triangle of
tau_hat_k^2 I - T_k / s_k^2 - G_hat_k G_hat_k' = 0. T_k is `control_covariance` written out
as a polynomial of (f, K_hat, L_tilde):

    T = alpha [d (K F)(K F)' + K diag(m) K'] + beta tr(d (K F)(K F)' + K diag(m) K') I + Q_G(f) + eps I,

with K = gain_scale K_hat, F = diag(std) L (6x6 block of the arc-start factor; the data L0 on
arc 1), m_b = d R_nav,b, alpha = ct (1 + s2 - s4), beta = ct s4.
"""
function add_slack_chance(c, n, scaling, seed, p, ff, gains, chol, tau, slack_factor, slack_scale)
    chance_data = [(k = a, w = p.psi * slack_scale[a], e2 = p.eps^2) for a in 1:n]
    @add_con(c, sqrt(ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2 + t.e2) + t.w * tau[t.k] for t in chance_data;
        lcon = -Inf, ucon = 1.0)

    rows = [(k = a, i = i, j = j) for a in 1:n for i in 1:NU for j in 1:i]
    row_of = Dict((r.k, r.i, r.j) => id for (id, r) in enumerate(rows))
    kappa = 1.0 ./ slack_scale .^ 2
    base = [merge((k = r.k, delta = Float64(r.i == r.j), c0 = (r.i == r.j) * p.jitter * kappa[r.k]), product_fields(r.i, r.j, NU))
            for r in rows]
    @add_con(c, slack_rows, t.delta * tau[t.k]^2 - t.c0 -
        (t.m1 * slack_factor[t.a1, t.k] * slack_factor[t.b1, t.k] + t.m2 * slack_factor[t.a2, t.k] * slack_factor[t.b2, t.k] +
         t.m3 * slack_factor[t.a3, t.k] * slack_factor[t.b3, t.k]) for t in base)

    alpha = p.ct * (1 + p.var_mag - p.var_point)
    beta = p.ct * p.var_point
    isdiag(p.nav_cov) || error("the slack chance formulation needs a diagonal navigation covariance")
    m = [p.d * p.nav_cov[b, b] for b in 1:NP]
    L0 = unpack_lower(scaling.initial_factor, NX)
    # (K F)_{q,col} = sum_row c_row K_hat[q,row] L[r_row]: weights and packed factor rows of column col
    kf_weights(a, col) = [row >= col ? scaling.std[row, a] : 0.0 for row in 1:NP]
    kf_index(col) = [packed_index(max(row, col), col) for row in 1:NP]
    kf_fields(a, col) = (w = kf_weights(a, col); r = kf_index(col);
        (c1 = w[1], r1 = r[1], c2 = w[2], r2 = r[2], c3 = w[3], r3 = r[3], c4 = w[4], r4 = r[4], c5 = w[5], r5 = r[5], c6 = w[6], r6 = r[6]))
    kf0_fields(col) = (w = [row >= col ? scaling.std[row, 1] * L0[row, col] : 0.0 for row in 1:NP];
        (c1 = w[1], c2 = w[2], c3 = w[3], c4 = w[4], c5 = w[5], c6 = w[6]))
    gs2(a) = seed.gain_scale[a]^2

    # -alpha d (K F)_i . (K F)_j and, on the diagonal, -beta d sum_q ||(K F)_q||^2 (arcs 2..n: factor variables)
    pair = [merge((id = row_of[(a, i, j)], a = a, k = a - 1, i = i, j = j, w = -kappa[a] * alpha * p.d * gs2(a)), kf_fields(a, col))
            for a in 2:n for i in 1:NU for j in 1:i for col in 1:NP]
    trace = [merge((id = row_of[(a, i, i)], a = a, k = a - 1, q = q, w = -kappa[a] * beta * p.d * gs2(a)), kf_fields(a, col))
             for a in 2:n for i in 1:NU for q in 1:NU for col in 1:NP]
    if !isempty(pair)
        @add_con!(c, slack_rows, t.id => t.w *
            (t.c1 * gains[t.i, 1, t.a] * chol[t.r1, t.k] + t.c2 * gains[t.i, 2, t.a] * chol[t.r2, t.k] +
             t.c3 * gains[t.i, 3, t.a] * chol[t.r3, t.k] + t.c4 * gains[t.i, 4, t.a] * chol[t.r4, t.k] +
             t.c5 * gains[t.i, 5, t.a] * chol[t.r5, t.k] + t.c6 * gains[t.i, 6, t.a] * chol[t.r6, t.k]) *
            (t.c1 * gains[t.j, 1, t.a] * chol[t.r1, t.k] + t.c2 * gains[t.j, 2, t.a] * chol[t.r2, t.k] +
             t.c3 * gains[t.j, 3, t.a] * chol[t.r3, t.k] + t.c4 * gains[t.j, 4, t.a] * chol[t.r4, t.k] +
             t.c5 * gains[t.j, 5, t.a] * chol[t.r5, t.k] + t.c6 * gains[t.j, 6, t.a] * chol[t.r6, t.k]) for t in pair)
        @add_con!(c, slack_rows, t.id => t.w *
            (t.c1 * gains[t.q, 1, t.a] * chol[t.r1, t.k] + t.c2 * gains[t.q, 2, t.a] * chol[t.r2, t.k] +
             t.c3 * gains[t.q, 3, t.a] * chol[t.r3, t.k] + t.c4 * gains[t.q, 4, t.a] * chol[t.r4, t.k] +
             t.c5 * gains[t.q, 5, t.a] * chol[t.r5, t.k] + t.c6 * gains[t.q, 6, t.a] * chol[t.r6, t.k])^2 for t in trace)
    end
    # the same on arc 1, whose factor L0 is data
    pair0 = [merge((id = row_of[(1, i, j)], i = i, j = j, w = -kappa[1] * alpha * p.d * gs2(1)), kf0_fields(col))
             for i in 1:NU for j in 1:i for col in 1:NP]
    trace0 = [merge((id = row_of[(1, i, i)], q = q, w = -kappa[1] * beta * p.d * gs2(1)), kf0_fields(col))
              for i in 1:NU for q in 1:NU for col in 1:NP]
    @add_con!(c, slack_rows, t.id => t.w *
        (t.c1 * gains[t.i, 1, 1] + t.c2 * gains[t.i, 2, 1] + t.c3 * gains[t.i, 3, 1] +
         t.c4 * gains[t.i, 4, 1] + t.c5 * gains[t.i, 5, 1] + t.c6 * gains[t.i, 6, 1]) *
        (t.c1 * gains[t.j, 1, 1] + t.c2 * gains[t.j, 2, 1] + t.c3 * gains[t.j, 3, 1] +
         t.c4 * gains[t.j, 4, 1] + t.c5 * gains[t.j, 5, 1] + t.c6 * gains[t.j, 6, 1]) for t in pair0)
    @add_con!(c, slack_rows, t.id => t.w *
        (t.c1 * gains[t.q, 1, 1] + t.c2 * gains[t.q, 2, 1] + t.c3 * gains[t.q, 3, 1] +
         t.c4 * gains[t.q, 4, 1] + t.c5 * gains[t.q, 5, 1] + t.c6 * gains[t.q, 6, 1])^2 for t in trace0)

    # navigation part: -alpha K diag(m) K' and, on the diagonal, -beta tr(K diag(m) K')
    nav_pair = [(id = row_of[(a, i, j)], a = a, i = i, j = j, b = b, w = -kappa[a] * alpha * gs2(a) * m[b])
                for a in 1:n for i in 1:NU for j in 1:i for b in 1:NP]
    nav_trace = [(id = row_of[(a, i, i)], a = a, q = q, b = b, w = -kappa[a] * beta * gs2(a) * m[b])
                 for a in 1:n for i in 1:NU for q in 1:NU for b in 1:NP]
    @add_con!(c, slack_rows, t.id => t.w * gains[t.i, t.b, t.a] * gains[t.j, t.b, t.a] for t in nav_pair)
    @add_con!(c, slack_rows, t.id => t.w * gains[t.q, t.b, t.a]^2 for t in nav_trace)

    # Gates execution error -Q_G(f) = -(s4 ||f||^2 I + (s2 - s4) f f')
    gates = [(id = row_of[(r.k, r.i, r.j)], k = r.k, i = r.i, j = r.j, w1 = -kappa[r.k] * p.var_point * (r.i == r.j),
              w2 = -kappa[r.k] * (p.var_mag - p.var_point)) for r in rows]
    @add_con!(c, slack_rows, t.id => t.w1 * (ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2) + t.w2 * ff[t.i, t.k] * ff[t.j, t.k]
        for t in gates)
    return c
end
