# Stochastic multiple-shooting NLP (the seeded TrajectoryNLP of the JAX script),
# built from a `StochasticLayout` shared by the single-phase cases and the
# multiphase Psyche case.
#
# Arc k maps node k to node k+1 (n arcs, n + 1 nodes). Variables: node means 7(n+1), the unit directions
# d_k 3C and scaled gains K_hat = K_tilde / gain_scale 18C of the C controlled arcs, seed-scaled node Cholesky
# factors of nodes 2..n+1 28n, the throttles Gamma_k in [0, 1] C (S_k = Gamma_k d_k in T_max units), then
# (step 4) the terminal margin factor 21 and the slack epigraph tau_hat C and G_hat 6C. Constraints: fixed node
# components (boundary rows), mean defects 7n, square-root covariance defects L_{k+1} - chol(V_k) 28n (the
# native part is linear; `ut_factor.jl`), (step 4) the terminal rows 21, the thrust chance rows C, (step 4) the
# slack rows 6C, the unit-direction rows ||d_k||^2 = 1 C, then case-specific rows. Bounds: node-mean bounds of
# the layout, nonnegative factor diagonals, (step 4) margin diagonal >= TERMINAL_MARGIN_FLOOR and slack factor
# diagonal >= CHANCE_SLACK_FLOOR / s_k.
#
# Step 3 (objective = :standard), arc weights w_k (w_P = state_covariance_weight, w = control_covariance_weight):
#   sum_controlled w_k (Gamma_k + w tr(T_k)) + sum_all w_k w_P tr(P_k[1:6,1:6]) / c^2 + w_N tr(G P_N G')/c^2,
# P_k the covariance at the start node of arc k, with tr(T_k) in the closed form of `control_covariance`:
#   tr(T) = (1 + s2 + 2 s4) ct tr(K M K') + (s2 + 2 s4) Gamma^2 ||d||^2 + 3 eps,
#   tr(K M K') = d tr(K P6 K') + d sum_qb K_qb^2 R_nav,b.
# Terms constant in the variables (the fixed initial covariance and 3 w eps per controlled arc) are left out of
# the NLP and returned as `objective_constant`. Chance rows Gamma_k + psi rho(T_k) <= 1 (`chance_oracle`).
# Step 4 (objective = :chance_sum): sum_k w_k (Gamma_k + psi s_k tau_hat_k) with the slack epigraph
# tau_hat_k^2 I - G_hat_k G_hat_k' - T_k / s_k^2 = 0 (`chance_slack_oracle`), chance rows
# Gamma_k + psi s_k tau_hat_k <= 1 and the terminal bound I - G P_N G' - M M' = 0.

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
    return StochasticLayout(collect(1:n), single_phase_arc_weights(ref), collect(ref.steps), boundary, zeros(NX, n), ones(NX, n),
        fill(-Inf, NX, n + 1), fill(Inf, NX, n + 1), [(collect(1:n), p)], p, Matrix(Diagonal(terminal_inverse_std(o))))
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
UT oracle layout of the arcs `arcs` (one group): local input indices of the add_eval
variable list (means, ff, gains, chol, throttle), fixed inputs and output rows of the
(mean_defects, cov_defects) blocks.
"""
function group_ut_layout(l::StochasticLayout, arcs, offsets, scaling::CovarianceScaling, seed::InitialGuess, inputs)
    n, m = n_arcs(l), length(arcs)
    theta_index = zeros(Int, inputs, m)
    theta_fixed = zeros(inputs, m)
    rows = zeros(Int, NUT_ROWS, m)
    for (a, k) in enumerate(arcs)
        theta_index[1:7, a] = offsets.means .+ 7(k-1) .+ (1:7)
        slot = l.slots[k]
        if slot > 0
            theta_index[8:10, a] = offsets.ff .+ 3(slot-1) .+ (1:3)
            theta_index[57, a] = offsets.throttle + slot          # throttle Gamma_k
            theta_index[11:28, a] = offsets.gains .+ 18(slot-1) .+ (1:18)
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
may append case-specific rows after the unit-direction rows and returns the core. `uts` holds the UT
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
    lvar_margin[collect(MTRI_DIAGONAL)] .= TERMINAL_MARGIN_FLOOR
    gs = seed.gain_scale[controlled]
    step4 = chance_sum(o)
    @add_var(c, means, 1:NX, 1:n+1; start = seed.means, lvar = l.mean_lower, uvar = l.mean_upper)
    # Gamma = ||S_seed|| with the seed direction reproduces the seed feedforward; Gamma >= 0 holds at every
    # iterate (no bound relaxation), so ||Gamma d|| = Gamma.
    S0 = seed.feedforward[:, controlled]
    gamma0 = [norm(S0[:, k]) for k in 1:nc]
    directions = reduce(hcat, [gamma0[k] > 0 ? S0[:, k] / gamma0[k] : [1.0, 0.0, 0.0] for k in 1:nc])
    @add_var(c, ff, 1:NU, 1:nc; start = directions)   # direction d_k (rows ||d_k||^2 = 1)
    @add_var(c, gains, 1:NU, 1:NP, 1:nc; start = seed.gains[:, :, controlled] ./ reshape(gs, 1, 1, :))
    @add_var(c, chol, 1:NL, 1:n; start = scaling.factors, lvar = lvar_chol)
    # right after the factors: the UT oracle's variable list (means, ff, gains, chol, throttle) stays contiguous
    @add_var(c, throttle, 1:nc; start = gamma0, lvar = zeros(nc), uvar = ones(nc))
    margin = nothing
    step4 && @add_var(c, margin, 1:NM; start = seed.terminal_margin, lvar = lvar_margin)
    tau = slack_factor = slack_scale = nothing
    if step4
        starts = [slack_oracle_start(T, CHANCE_SLACK_FLOOR) for T in start_control_covariances(seed, scaling, p, controlled)]
        slack_scale = [first(s) for s in starts]
        lvar_slack = fill(-Inf, NSLACK, nc)
        for k in 1:nc, r in 1:NSLACK
            SLACK_ROWS[r] == SLACK_COLS[r] && (lvar_slack[r, k] = CHANCE_SLACK_FLOOR / slack_scale[k])
        end
        @add_var(c, tau, 1:nc; start = [s[2] for s in starts], lvar = zeros(nc))
        @add_var(c, slack_factor, 1:NSLACK, 1:nc; start = reduce(hcat, [s[3] for s in starts]), lvar = lvar_slack)
    end

    # ------------------------------------------------------------ objective
    constant = 0.0
    if step4
        # sum_k w_k (Gamma_k + psi s_k tau_hat_k): the chance left-hand sides with the epigraph tau_hat
        cost_data = [(k = k, w = l.weights[a], ps = p.psi * slack_scale[k]) for (k, a) in enumerate(controlled)]
        @add_obj(c, t.w * (throttle[t.k] + t.ps * tau[t.k]) for t in cost_data)
    else
        @add_obj(c, t.w * throttle[t.k] for t in [(k = l.slots[a], w = l.weights[a]) for a in controlled])
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
        @add_obj(c, t.w * throttle[t.k]^2 * (ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2)
            for t in [(k = l.slots[a], w = cf * l.weights[a]) for a in controlled])
        # w_N tr(G P_N G')/c^2 = w_N/c^2 sum_{i, col} (sum_k M_ik L_N[k, col])^2 with M = G S_N
        M = l.terminal_transform * node_scale(scaling, n + 1)[1:NP, 1:NP]
        wN = TERMINAL_COVARIANCE_WEIGHT / BRYSON_SIGMA_FACTOR^2
        terminal_cost = [begin
            row = [k >= col ? M[i, k] : 0.0 for k in 1:NP]
            index = [packed_index(max(k, col), col) for k in 1:NP]
            (w = wN, c1 = row[1], r1 = index[1], c2 = row[2], r2 = index[2], c3 = row[3], r3 = index[3],
             c4 = row[4], r4 = index[4], c5 = row[5], r5 = index[5], c6 = row[6], r6 = index[6])
        end for i in 1:NP for col in 1:NP if any(!iszero, M[i, col:NP])]
        @add_obj(c, t.w * (t.c1 * chol[t.r1, n] + t.c2 * chol[t.r2, n] + t.c3 * chol[t.r3, n] +
            t.c4 * chol[t.r4, n] + t.c5 * chol[t.r5, n] + t.c6 * chol[t.r6, n])^2 for t in terminal_cost)
        constant += sum(o.control_covariance_weight * 3p.jitter * l.weights[a] for a in controlled; init = 0.0)
    end

    # ---------------------------------------------------------- constraints
    @add_con(c, means[t.i, t.node] - t.value for t in l.boundary)
    # Native parts of the defect rows; the UT oracles below add the predicted moments.
    # The origin translations of switch arcs enter the rows, not their bounds.
    @add_con(c, mean_defects, means[t.i, t.k+1] - t.rhs for t in [(i = i, k = k, rhs = l.defect_rhs[i, k]) for k in 1:n for i in 1:NX])
    @add_con(c, cov_defects, chol[t.r, t.k] for t in [(k = a, r = r) for a in 1:n for r in 1:NL])
    # Terminal covariance bound G P_N G' <= I (matrix inequality, step 4):  I - G P_N G' - M M' = 0
    G = l.terminal_transform
    std_N = scaling.std[:, n+1]
    if !step4
    elseif isdiag(G) && !scaling.full
        inverse_target = diag(G)
        terminal_data = [begin
            i, j = LTRI_ROWS[r], LTRI_COLS[r]
            merge((delta = Float64(i == j), w = std_N[i] * std_N[j] * inverse_target[i] * inverse_target[j]),
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
        terminal_data = [merge((delta = Float64(LTRI_ROWS[r] == LTRI_COLS[r]),),
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

    # --------------------------------------------------------------- oracles
    # add_eval's variable list starts at the first variable, so its local indices are global.
    offsets = (; means = means.offset, ff = ff.offset, gains = gains.offset, chol = chol.offset, throttle = throttle.offset)
    offsets.means == 0 && offsets.ff == NX * (n + 1) || error("unexpected variable layout")
    offsets.throttle != chol.offset + NL * n && error("the throttle must follow the factors")
    sort(reduce(vcat, first.(l.groups))) == 1:n || error("the UT groups must cover every arc once")
    uts = UTOracle[]
    for (arcs, params) in l.groups
        ut = ut_oracle(group_ut_layout(l, arcs, offsets, scaling, seed, ntheta(params)), params, backend)
        fu = ut_factor_oracle(ut)
        c, _ = ExaModels.add_eval(c, (mean_defects, cov_defects), (means, ff, gains, chol, throttle),
            (out, x) -> ut_factor_values!(out, x, fu);
            jac! = (vals, x) -> ut_factor_jacobian!(vals, x, fu),
            hess! = (vals, x, y) -> ut_factor_hessian!(vals, x, y, fu),
            jac_structure! = (r, cc) -> (append!(r, ut.structure.jac_rows); append!(cc, ut.structure.jac_cols)),
            hess_structure! = (r, cc) -> (append!(r, ut.structure.hess_rows); append!(cc, ut.structure.hess_cols)))
        push!(uts, ut)
    end
    nvar = NX * (n + 1) + (NU + NU * NP) * nc + NL * n + nc + (step4 ? NM + 7nc : 0)
    layout = chance_layout(l.slots[controlled], controlled, offsets, scaling, gs)
    if step4
        chance_data = [(k = k, w = p.psi * slack_scale[k]) for k in 1:nc]
        @add_con(c, throttle[t.k] + t.w * tau[t.k] for t in chance_data; lcon = -Inf, ucon = 1.0)
        c = ExaModels.constraint(c, chance_slack_oracle(layout, nvar, p, backend, tau.offset, slack_factor.offset,
            1.0 ./ slack_scale .^ 2))
    else
        c = ExaModels.constraint(c, chance_oracle(layout, nvar, p, backend))
    end
    @add_con(c, unit_direction, ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2 - 1 for t in [(k = k,) for k in 1:nc];
        lcon = zeros(nc), ucon = zeros(nc))
    blocks = (; means, ff, gains, chol, margin, throttle)
    extra_constraints === nothing || (c = extra_constraints(c, blocks, nvar))
    model = ExaModel(c)
    model.meta.nvar == nvar || error("variable count $(model.meta.nvar) differs from the expected $nvar")
    ncon = length(l.boundary) + (NX + NL) * n + nc + nc + (step4 ? NM + 6nc : 0)
    extra_constraints === nothing && model.meta.ncon != ncon &&
        error("constraint count $(model.meta.ncon) differs from the expected $ncon")
    chance_slack = step4 ? (; tau, factor = slack_factor, scale = slack_scale) : nothing
    return (; model, means, ff, gains, chol, margin, throttle, chance_slack, objective_constant = step4 ? 0.0 : constant,
        uts, ut = first(uts), controlled)
end

"The stochastic NLP of a single-phase case."
build_stochastic_model(case::CaseData, o::Options, ref::ReferenceTraj, seed::InitialGuess,
        scaling::CovarianceScaling, p::UTParams, backend) =
    build_stochastic_nlp(single_phase_layout(case, o, ref, p), o, seed, scaling, backend)
