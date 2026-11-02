# Psyche stochastic NLP: the shared `build_stochastic_nlp` with the multiphase layout and
# the Mars clearance chance constraint.
#
# Layout: arcs weighted by N dt_k / (t_f - t_0); feedforward and gains on the 101 cruise
# arcs only; boundary rows for the fixed Earth SOI exit (7), Mars SOI entry (6) and arrival
# (6) states; the Mars epochs fixed at the energy reoptimisation's values, so every interval
# duration and both origin translations are constants (in the switch arcs' mean-defect rows);
# mean mass >= the mass floor; two UT parameter groups (solar cruises with Gates execution
# error, ballistic Mars without thrust); terminal bound diag(sp^2 I3, sv^2 I3).
# Last row: the B-plane clearance of the closest-approach node (risk 0.001, 500 km).

const PSYCHE_FLYBY_RISK = 0.001

function psyche_layout(d::PsycheData, u::PsycheUncertainty, params, ref)
    n = d.n
    ratio = reduce(hcat, [d.phase_scales[:, d.phase[k]] ./ d.node_scales[:, k+1] for k in 1:n])
    rhs = reduce(hcat, [switch_translation(d, k, ref.days) for k in 1:n])
    lower, upper = fill(-Inf, NX, n + 1), fill(Inf, NX, n + 1)
    lower[7, :] .= d.mass_floor_kg ./ d.node_scales[7, :]
    groups = [(findall(!=(2), d.phase), params[1]), (findall(==(2), d.phase), params[2])]
    return StochasticLayout(copy(d.slots), arc_weights(d, ref.days), collect(ref.steps), psyche_boundary(d), rhs, ratio,
        lower, upper, groups, params[1], copy(u.terminal_transform))
end

"""
Clearance of the closest-approach node from its mean (node units, inputs 1..7) and the first
21 entries of its normalized factor (inputs 8..28): P_SI = (T F)(T F)' with F the scaled factor.
"""
struct PsycheClearance
    scale::SVector{7,Float64}             # node units -> SI
    transform::SMatrix{6,6,Float64,36}    # D S_ca (6 x 6 blocks): factor variable -> SI
    mu::Float64
    rho::Float64
    chi::Float64
end

@inline function (e::PsycheClearance)(v)
    z = SVector{6}(ntuple(i -> v[i] * e.scale[i], Val(6)))
    LL = position_velocity_covariance(SVector{7}(ntuple(_ -> 1.0, Val(7))), SVector{21}(ntuple(i -> v[7+i], Val(21))))
    P = e.transform * LL * e.transform'
    r = SVector(z[1], z[2], z[3])
    return encounter_clearance(z, P, -e.mu * r / norm(r)^3, e.mu, e.rho, e.chi)
end

const NCLEARANCE = 28
const NCLEARANCE_PAIRS = NCLEARANCE * (NCLEARANCE + 1) ÷ 2

@inline clearance_inputs(x, index) = SVector{NCLEARANCE}(ntuple(i -> @inbounds(x[index[i]]), Val(NCLEARANCE)))

@kernel function clearance_value_kernel!(out, @Const(x), @Const(index), e)
    i = @index(Global, Linear)
    @inbounds out[i] = e(clearance_inputs(x, index))
end

@kernel function clearance_jacobian_kernel!(jac, @Const(x), @Const(index), e)
    q = @index(Global, Linear)
    v = clearance_inputs(x, index)
    @inbounds jac[q] = ForwardDiff.partials(e(SVector{NCLEARANCE}(ntuple(j -> single_seed(v[j], j == q), Val(NCLEARANCE)))))[1]
end

@kernel function clearance_hessian_kernel!(hess, @Const(x), @Const(y), @Const(index), @Const(pair_p), @Const(pair_q), e)
    pair = @index(Global, Linear)
    pp, qq = @inbounds(pair_p[pair]), @inbounds(pair_q[pair])
    v = clearance_inputs(x, index)
    value = e(SVector{NCLEARANCE}(ntuple(j -> pair_seed(v[j], pp == j, qq == j), Val(NCLEARANCE))))
    @inbounds hess[pair] = y[1] * pair_value(value)
end

"Clearance row on `backend`: exact first and second derivatives from single and pair-seeded duals."
function clearance_oracle(e::PsycheClearance, index::Vector{Int}, nvar, backend)
    pair_p = [pp for pp in 1:NCLEARANCE for _ in 1:pp]
    pair_q = [qq for pp in 1:NCLEARANCE for qq in 1:pp]
    device(v) = ExaModels.convert_array(v, backend)
    ids, dp, dq = device(index), device(pair_p), device(pair_q)
    return ExaModels.VectorNonlinearOracle(; nvar, ncon = 1, jac_rows = ones(Int, NCLEARANCE), jac_cols = index,
        hess_rows = [max(index[p], index[q]) for (p, q) in zip(pair_p, pair_q)],
        hess_cols = [min(index[p], index[q]) for (p, q) in zip(pair_p, pair_q)],
        lcon = device([0.0]), ucon = device([Inf]),
        f! = (out, x) -> launch!(clearance_value_kernel!, 1, out, x, ids, e),
        jac! = (vals, x) -> launch!(clearance_jacobian_kernel!, NCLEARANCE, vals, x, ids, e),
        hess! = (vals, x, y) -> launch!(clearance_hessian_kernel!, NCLEARANCE_PAIRS, vals, x, y, ids, dp, dq, e))
end

psyche_clearance(d::PsycheData, u::PsycheUncertainty, scaling::CovarianceScaling) =
    PsycheClearance(SVector{7}(d.node_scales[:, d.ca_node]),
        SMatrix{6,6}(u.factor_si[1:NP, 1:NP] * node_scale(scaling, d.ca_node)[1:NP, 1:NP]),
        d.mu[2], d.mars_radius_m + PSYCHE_MINIMUM_MARS_ALTITUDE_M, sqrt(-2log(PSYCHE_FLYBY_RISK)))

function psyche_stochastic_model(d::PsycheData, o::Options, u::PsycheUncertainty, params, ref, seed::InitialGuess,
        scaling::CovarianceScaling, backend)
    layout = psyche_layout(d, u, params, ref)
    clearance = psyche_clearance(d, u, scaling)
    extra(c, blocks, nvar) = ExaModels.constraint(c, clearance_oracle(clearance,
        vcat(blocks.means.offset .+ 7(d.ca_node-1) .+ (1:NX), blocks.chol.offset .+ 28(d.ca_node-2) .+ (1:21)), nvar, backend))
    blocks = build_stochastic_nlp(layout, o, seed, scaling, backend; extra_constraints = extra)
    return merge(blocks, (; layout, clearance))
end
