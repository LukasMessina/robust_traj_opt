# Fuel reoptimisation of the deterministic Psyche reference on the stochastic mesh (CPU, exact derivatives).
# It restores dynamic consistency with one XMDS2 RK9 step per interval and fixes the Mars epochs used by the
# stochastic NLP.
#
# Variables: node means, the deterministic optimisation's throttle Gamma_k in [0, 1] and direction d_k of the
# powered arcs (thrust U = Gamma d, mass flow T_max Gamma, no smoothing) and the Mars inbound/outbound days.
# Objective: the fuel 1 - m_f / m_wet (the constant left out).
# Constraints: fixed Earth SOI exit (7), Mars SOI entry (6) and arrival (6) states, shooting
# defects (with the ephemeris translations at both SOI crossings), ||d_k||^2 = 1, Mars
# altitude >= 500 km at the Mars nodes, and the events: Mars SOI exit radius, closest-approach
# r.v = 0, outward exit crossing, 0.05..10 day Mars passage, q = v.v + r.a > 0 and C3 > 0
# (a hyperbolic closest-approach minimum).

const PSYCHE_MINIMUM_MARS_ALTITUDE_M = 500e3

"Inputs of the event rows: closest-approach mean (Mars), Mars-exit mean (Sun), days."
function psyche_events(d::PsycheData, v)
    mars, sun = d.node_scales[:, d.ca_node], d.node_scales[:, d.exit_node]
    r = SVector{3}(ntuple(i -> v[i] * mars[i], Val(3)))
    w = SVector{3}(ntuple(i -> v[3+i] * mars[3+i], Val(3)))
    planet = d.planet(mars_exit_epoch(d, (v[15], v[16])))
    re = SVector{3}(ntuple(i -> v[7+i] * sun[i] - planet[i], Val(3)))
    we = SVector{3}(ntuple(i -> v[10+i] * sun[3+i] - planet[3+i], Val(3)))
    mu = d.mu[2]
    # Ballistic Mars: the closest-approach acceleration is Mars gravity alone.
    acceleration = -mu * r / norm(r)^3
    return SVector(norm(re) / d.mars_soi_m - 1, dot(r, w) / (mars[1] * mars[4]), dot(re, we) / (d.mars_soi_m * mars[4]),
        v[15] + v[16], (dot(w, w) + dot(r, acceleration)) / 1e6, (dot(w, w) - 2mu / norm(r)) / 1e6)
end

const PSYCHE_EVENT_LOWER = [0.0, 0.0, 0.0, 0.05, 0.0, 0.0]
const PSYCHE_EVENT_UPPER = [0.0, 0.0, Inf, 10.0, Inf, Inf]

"Event rows of a mean trajectory (SOI exit, CA, outward crossing, duration, q, C3)."
function psyche_event_report(d::PsycheData, means, days)
    e = psyche_events(d, vcat(means[:, d.ca_node], means[:, d.exit_node], days))
    return Dict{String,Any}("days" => collect(Float64.(days)), "soi_equality" => e[1], "ca_equality" => e[2],
        "outward_crossing" => e[3], "mars_duration_days" => e[4], "q_scaled" => e[5], "c3_scaled" => e[6],
        "soi_radius_error_m" => e[1] * d.mars_soi_m, "strict_hyperbolic_minimum" => e[5] > 0 && e[6] > 0,
        "closest_approach_epoch_et" => d.epochs[2] + 86400 * days[1], "mars_exit_epoch_et" => mars_exit_epoch(d, days))
end

"Lower-triangle (row >= col) entries of the Hessian of `y' f(v)` over the active inputs `index` (0: data)."
function lower_entries!(vals, offset, H, index)
    position = offset
    for p in eachindex(index), q in 1:p
        (index[p] == 0 || index[q] == 0) && continue
        vals[position += 1] = H[p, q]
    end
    return position
end

function lower_structure!(rows, cols, index)
    for p in eachindex(index), q in 1:p
        (index[p] == 0 || index[q] == 0) && continue
        push!(rows, max(index[p], index[q])); push!(cols, min(index[p], index[q]))
    end
end

function psyche_nominal_model(d::PsycheData, o::Options, params)
    n = d.n
    powered = findall(>(0), d.slots)
    np = length(powered)
    lower, upper = fill(-Inf, NX, n + 1), fill(Inf, NX, n + 1)
    lower[7, :] .= d.mass_floor_kg ./ d.node_scales[7, :]
    c = ExaCore(Float64)
    @add_var(c, means, 1:NX, 1:n+1; start = d.seed_states, lvar = lower, uvar = upper)
    # the deterministic seed convention: every nonzero thrust kept, zero thrust with direction e1
    S0 = d.seed_controls[:, powered]
    gamma0 = [norm(S0[:, k]) for k in 1:np]
    @add_var(c, ff, 1:NU, 1:np; start = reduce(hcat, [gamma0[k] > 0 ? S0[:, k] / gamma0[k] : [1.0, 0.0, 0.0] for k in 1:np]))
    @add_var(c, throttle, 1:np; start = gamma0, lvar = zeros(np), uvar = ones(np))
    @add_var(c, days, 1:2; start = d.seed_days, lvar = fill(1e-9, 2), uvar = fill(10.0, 2))
    # the deterministic optimisation's fuel objective 1 - m_f / m_wet (the constant 1 left out)
    @add_obj(c, -t.s * means[7, t.node] for t in [(node = n + 1, s = d.node_scales[7, n+1] / d.wet_mass_kg)])
    boundary = psyche_boundary(d)
    @add_con(c, means[t.i, t.node] - t.value for t in boundary)
    @add_con(c, defects, means[t.i, t.k+1] for t in [(i = i, k = k) for k in 1:n for i in 1:NX])
    @add_con(c, ff[1, t.k]^2 + ff[2, t.k]^2 + ff[3, t.k]^2 - 1 for t in [(k = k,) for k in 1:np])
    radius = d.mars_radius_m + PSYCHE_MINIMUM_MARS_ALTITUDE_M
    surfaces = [(node = node, r2 = (radius / d.node_scales[1, node])^2) for node in 1:n+1 if d.node_origin[node] == 2]
    @add_con(c, means[1, t.node]^2 + means[2, t.node]^2 + means[3, t.node]^2 - t.r2 for t in surfaces; lcon = 0.0, ucon = Inf)

    # Shooting defects means[k+1] - end_k(theta_k): exact first and second derivatives by ForwardDiff.
    # theta_k = (mean 7, direction 3, throttle, days 2).
    index = [vcat(means.offset .+ 7(k-1) .+ (1:7), d.slots[k] == 0 ? zeros(Int, 3) : ff.offset .+ 3(d.slots[k]-1) .+ (1:3),
                  [d.slots[k] == 0 ? 0 : throttle.offset + d.slots[k]], days.offset .+ (1:2)) for k in 1:n]
    ninputs = 13
    arc_end = psyche_arc_end_throttle
    input(x, k, q) = index[k][q] == 0 ? 0.0 : x[index[k][q]]
    load(x, k) = SVector{13}(ntuple(q -> input(x, k, q), Val(13)))
    jac_rows, jac_cols, hess_rows, hess_cols = Int[], Int[], Int[], Int[]
    jac_offsets, hess_offsets = zeros(Int, n), zeros(Int, n)
    for k in 1:n
        jac_offsets[k], hess_offsets[k] = length(jac_rows), length(hess_rows)
        for i in 1:NX, q in 1:ninputs
            index[k][q] == 0 && continue
            push!(jac_rows, 7(k-1) + i); push!(jac_cols, index[k][q])
        end
        lower_structure!(hess_rows, hess_cols, index[k])
    end
    c, _ = ExaModels.add_eval(c, (defects,), (means, ff, throttle, days),
        function (out, x)
            Threads.@threads for k in 1:n
                out[7(k-1) .+ (1:7)] = -arc_end(d, params, k, load(x, k))
            end
        end;
        jac! = function (vals, x)
            Threads.@threads for k in 1:n
                J = ForwardDiff.jacobian(v -> -arc_end(d, params, k, v), load(x, k))
                position = jac_offsets[k]
                for i in 1:NX, q in 1:ninputs
                    index[k][q] == 0 && continue
                    vals[position += 1] = J[i, q]
                end
            end
        end,
        hess! = function (vals, x, y)
            Threads.@threads for k in 1:n
                g = SVector{7}(ntuple(i -> -y[7(k-1)+i], Val(7)))
                H = ForwardDiff.hessian(v -> dot(g, arc_end(d, params, k, v)), load(x, k))
                lower_entries!(vals, hess_offsets[k], H, index[k])
            end
        end,
        jac_structure! = (r, cc) -> (append!(r, jac_rows); append!(cc, jac_cols)),
        hess_structure! = (r, cc) -> (append!(r, hess_rows); append!(cc, hess_cols)))

    nvar = NX * (n + 1) + NU * np + np + 2
    event_index = vcat(means.offset .+ 7(d.ca_node-1) .+ (1:7), means.offset .+ 7(d.exit_node-1) .+ (1:7),
        days.offset .+ (1:2))
    c = ExaModels.constraint(c, psyche_cpu_oracle(nvar, event_index, v -> psyche_events(d, v), 6,
        PSYCHE_EVENT_LOWER, PSYCHE_EVENT_UPPER))
    return (; model = ExaModel(c), means, ff, throttle, days)
end

"Small CPU oracle of `f(v)` (`rows` values) on the global inputs `index`, with ForwardDiff derivatives."
function psyche_cpu_oracle(nvar, index, f, rows, lower, upper)
    m = length(index)
    jac_rows = [r for r in 1:rows for _ in 1:m]
    jac_cols = [index[q] for _ in 1:rows for q in 1:m]
    hess_rows, hess_cols = Int[], Int[]
    lower_structure!(hess_rows, hess_cols, index)
    load(x) = [x[i] for i in index]
    return ExaModels.VectorNonlinearOracle(; nvar, ncon = rows, jac_rows, jac_cols, hess_rows, hess_cols,
        lcon = lower, ucon = upper,
        f! = (out, x) -> (out .= f(load(x))),
        jac! = (vals, x) -> (vals .= vec(permutedims(ForwardDiff.jacobian(f, load(x))))),
        hess! = (vals, x, y) -> lower_entries!(vals, 0, ForwardDiff.hessian(v -> dot(y, f(v)), load(x)), index))
end

"Fixed Earth SOI exit (state and mass), Mars SOI entry and arrival states, in node units."
psyche_boundary(d::PsycheData) =
    [[(node = 1, i = i, value = d.x0[i] / d.node_scales[i, 1]) for i in 1:NX];
     [(node = d.entry_node, i = i, value = d.xm[i] / d.node_scales[i, d.entry_node]) for i in 1:NP];
     [(node = d.n + 1, i = i, value = d.xf[i] / d.node_scales[i, d.n+1]) for i in 1:NP]]

"Fuel reoptimisation on the CPU; returns the reference (means, feedforward of every arc, days) and a report."
function psyche_reoptimize(d::PsycheData, o::Options, params; output_file = "")
    built = psyche_nominal_model(d, o, params)
    settings = psyche_nominal_solver_options(o; output_file)
    guard = settings.intermediate_callback
    stats, solver = run_madnlp(built.model, settings)
    means = Array(ExaModels.solution(stats, built.means))
    feedforward = zeros(NU, d.n)
    feedforward[:, d.slots .> 0] = Array(ExaModels.solution(stats, built.ff))
    feedforward[:, d.slots .> 0] .*= Array(ExaModels.solution(stats, built.throttle))'
    days = Array(ExaModels.solution(stats, built.days))
    report = Dict{String,Any}("status" => string(stats.status), "iterations" => stats.iter,
        "objective" => stats.objective, "max_constraint_violation" => original_violation(built.model, stats.solution),
        "restoration_iterations" => guard.restoration_iterations, "longest_restoration_run" => guard.max_consecutive,
        "objective_kind" => "fuel",
        "energy_n2_s" => sum(0.5 * d.thrust_max_n^2 * arc_duration(d, k, days) * sum(abs2, feedforward[:, k]) for k in 1:d.n),
        "propellant_kg" => means[7, 1] * d.node_scales[7, 1] - means[7, end] * d.node_scales[7, end],
        "minimum_mars_altitude_m" => PSYCHE_MINIMUM_MARS_ALTITUDE_M, "substeps_per_interval" => 1,
        "events" => psyche_event_report(d, means, days))
    stats.status == MadNLP.SOLVE_SUCCEEDED || error("the Psyche fuel reoptimisation did not converge: $(stats.status)")
    return (; means, feedforward, days, steps = arc_steps(d, days)), report
end
