# Psyche mission data: the mesh, events, boundary states and Mars ephemeris of the
# deterministic energy solution, exported by verification.export psyche, and the
# uncertainty model of a run. SI units, epochs in TDB seconds past J2000.
#
# Arc k maps node k to node k+1. Phases: 1 first solar cruise, 2 ballistic Mars passage
# (Mars-centred, no thrust), 3 final solar cruise. Each phase propagates in its own units
# (`phase_scales`, `time_scales`, unit gravitational parameter); a node's mean is in the
# units of its origin (`node_scales`). The two switch arcs end at the Mars SOI entry node
# (Sun to Mars) and at the Mars SOI exit node (Mars to Sun). Durations are affine in the
# Mars inbound and outbound days: dt_k = c1 + c2 d_in + c3 d_out.

import SHA

"Stored Mars ephemeris: cubic pieces of position and velocity in days from `origin` (heliocentric J2000)."
struct PsychePlanet
    knots::Vector{Float64}
    coefficients::Array{Float64,3}   # 4 x pieces x 6
    origin::Float64
    lo::Float64
    hi::Float64
end

"Mars state relative to the Sun at epoch `et` from the stored pieces (generic: `et` may be a dual number)."
function (planet::PsychePlanet)(et)
    day = (et - planet.origin) / 86400.0
    knots = planet.knots
    left, right = 1, length(knots)
    while left < right
        middle = (left + right + 1) ÷ 2
        if knots[middle] <= day
            left = middle
        else
            right = middle - 1
        end
    end
    index = min(left, length(knots) - 1)
    offset = day - knots[index]
    valid = planet.lo <= et <= planet.hi
    return SVector{6}(ntuple(Val(6)) do i
        c = view(planet.coefficients, :, index, i)
        value = ((c[1] * offset + c[2]) * offset + c[3]) * offset + c[4]
        valid ? value : value * NaN
    end)
end

struct PsycheData
    path::String
    sha256::String
    n::Int                            # intervals; nodes n + 1
    phase::Vector{Int}
    slots::Vector{Int}                # control slot of each arc (0 on Mars)
    duration::Matrix{Float64}         # 3 x n [s, s/day, s/day]
    switch::Vector{Bool}
    node_scales::Matrix{Float64}      # 7 x (n+1)
    node_origin::Vector{Int}          # phase whose origin and units each node uses
    phase_scales::Matrix{Float64}     # 7 x 3
    time_scales::Vector{Float64}
    mu::Vector{Float64}               # gravitational parameter of each phase's central body
    epochs::Vector{Float64}           # Earth SOI exit, Mars SOI entry, Psyche arrival
    x0::Vector{Float64}               # Sun-relative state and mass at Earth SOI exit
    xm::Vector{Float64}               # Mars-relative state at Mars SOI entry
    xf::Vector{Float64}               # Sun-relative asteroid state at arrival
    entry_node::Int
    exit_node::Int
    ca_node::Int
    thrust_max_n::Float64
    isp_s::Float64
    wet_mass_kg::Float64
    mass_floor_kg::Float64
    mars_radius_m::Float64
    mars_soi_m::Float64
    planet::PsychePlanet
    seed_states::Matrix{Float64}      # 7 x (n+1), node units
    seed_controls::Matrix{Float64}    # 3 x n, thrust / T_max
    seed_days::Vector{Float64}
    launch_covariance_si::Matrix{Float64}
    mesh_counts::Vector{Int}
    reference_mesh_counts::Vector{Int}
    mesh_refinement::Float64
    cruise_subdivision::Int           # solar-cruise intervals split into this many pieces after the refinement
    reference_path::String            # deterministic reference of the export, relative to the repository root
end

npz_string(bytes) = String(Vector{UInt8}(bytes))
npz_scalar(v) = v isa Number ? v : only(v)

function load_psyche_data(path::AbstractString)
    isfile(path) || error("Export the mission arrays first: python -m verification.export psyche " *
        "--output $path")
    a = NPZ.npzread(path)
    get(a, "ballistic_mars", 0) == 1 || error("regenerate the mission inputs from the ballistic-Mars energy solution")
    reference = joinpath(REPOSITORY_ROOT, npz_string(a["reference_path_utf8"]))
    isfile(reference) || error("the deterministic Psyche reference is missing: $reference")
    bytes2hex(SHA.sha256(read(reference))) == npz_string(a["reference_sha256_ascii"]) ||
        error("the deterministic Psyche reference changed; regenerate the mission input export")
    n = size(a["seed_controls"], 2)
    Int.(a["arc_start"]) == 1:n && Int.(a["arc_end"]) == 2:n+1 ||
        error("Psyche arcs must be consecutive shooting intervals")
    phase = Int.(a["arc_phase"])
    slots = max.(Int.(a["arc_control_index"]), 0)
    slots[phase .== 2] .= 0
    d = PsycheData(path, bytes2hex(SHA.sha256(read(path))), n, phase, slots, Float64.(a["arc_duration"]),
        Bool.(a["arc_switch"] .!= 0), Float64.(a["node_scales"]), Int.(a["node_origins"]),
        Float64.(a["phase_state_scales"]), Float64.(a["phase_time_scales"]), Float64.(a["phase_gravitational_parameters"]),
        Float64.(a["fixed_epochs"]), Float64.(a["x0"]), Float64.(a["xm"]), Float64.(a["xf"]),
        Int(npz_scalar(a["mars_entry_node"])), Int(npz_scalar(a["mars_exit_node"])),
        Int(npz_scalar(a["closest_approach_node"])),
        Float64(npz_scalar(a["thrust_max_n"])), Float64(npz_scalar(a["isp_s"])), Float64(npz_scalar(a["wet_mass_kg"])),
        Float64(npz_scalar(a["mass_floor_kg"])), Float64(npz_scalar(a["mars_radius_m"])),
        Float64(npz_scalar(a["mars_soi_m"])),
        PsychePlanet(Float64.(a["ephemeris_knots"]), Float64.(a["ephemeris_coefficients"]),
            Float64(npz_scalar(a["ephemeris_origin_et"])), Float64.(a["ephemeris_valid_interval"])...),
        Float64.(a["seed_states"]), Float64.(a["seed_controls"]), Float64.(a["seed_durations"]),
        Float64.(a["initial_covariance_si"]), Int.(a["phase_mesh_counts"]), Int.(a["reference_phase_mesh_counts"]),
        Float64(npz_scalar(get(a, "mesh_refinement", 1.0))), Int(npz_scalar(get(a, "cruise_subdivision", 1))),
        npz_string(a["reference_path_utf8"]))
    # Every arc starts at a node in its own phase's origin and units.
    all(k -> d.node_origin[k] == d.phase[k] && d.node_scales[:, k] == d.phase_scales[:, d.phase[k]], 1:n) ||
        error("an arc does not start in its own phase's units")
    d.phase_scales[:, 1] == d.phase_scales[:, 3] && d.time_scales[1] == d.time_scales[3] ||
        error("the two solar cruises must share their units")
    count(d.switch) == 2 && findall(d.switch) == [d.entry_node - 1, d.exit_node - 1] ||
        error("unexpected Mars SOI switch arcs")
    d.entry_node < d.ca_node < d.exit_node || error("the closest-approach node must lie inside the Mars passage")
    sort(unique(slots[slots .> 0])) == 1:count(>(0), slots) && issorted(slots[slots .> 0]) ||
        error("control slots must number the powered arcs in order")
    return d
end

"Duration [s] of arc k for the Mars inbound and outbound `days`."
@inline arc_duration(d::PsycheData, k, days) = d.duration[1, k] + d.duration[2, k] * days[1] + d.duration[3, k] * days[2]

"Propagation time of every arc in its phase's units."
arc_steps(d::PsycheData, days) = [arc_duration(d, k, days) / d.time_scales[d.phase[k]] for k in 1:d.n]

"Cost weights N dt_k / (t_f - t_0): unit weights on a uniform mesh."
arc_weights(d::PsycheData, days) = [d.n * arc_duration(d, k, days) for k in 1:d.n] ./ (d.epochs[3] - d.epochs[1])

"Epoch of the Mars SOI exit."
mars_exit_epoch(d::PsycheData, days) = d.epochs[2] + 86400.0 * (days[1] + days[2])

"""
Uncertainty model of a run. The initial covariance P0 at Earth SOI exit is the launch
covariance of the export (C3, RA, DEC 3-sigma bounds mapped to the asymptotic velocity,
correlations kept) with independent position and mass dispersions from the options.
Covariances are normalized by the initial standard deviations D = diag(sqrt(diag P0)) at every
node, whatever its origin (a change of origin at a fixed epoch leaves them unchanged):
P~ = D^-1 P D^-1. The NLP scales each node further by the full Cholesky factor of its seed
covariance (`covariance_scaling(...; full = true)`).
"""
struct PsycheUncertainty
    covariance_scale::Vector{Float64}        # D
    initial_covariance::Matrix{Float64}      # normalized (the correlation matrix of P0)
    initial_covariance_si::Matrix{Float64}
    factor_si::Matrix{Float64}               # D as a matrix: SI covariance D P~ D
    navigation_covariance::Vector{Float64}   # normalized variances
    terminal_std_si::Vector{Float64}
    terminal_transform::Matrix{Float64}      # G: terminal bound I - G P~_N G' >= 0 (6 x 6, diagonal)
end

function psyche_uncertainty(d::PsycheData, o::Options)
    P = zeros(NX, NX)
    P[1:3, 1:3] = (1000 * o.initial_position_std_km)^2 * I(3)
    P[4:6, 4:6] = d.launch_covariance_si[4:6, 4:6]
    P[7, 7] = o.initial_mass_std_kg^2
    scale = sqrt.(diag(P))
    navigation = [fill(1000 * o.navigation_position_std_km, 3); fill(1000 * o.navigation_velocity_std_km_s, 3)]
    terminal = [fill(1000 * o.psyche_terminal_position_std_km, 3); fill(1000 * o.psyche_terminal_velocity_std_km_s, 3)]
    return PsycheUncertainty(scale, P ./ (scale * scale'), P, Matrix(Diagonal(scale)), (navigation ./ scale[1:6]) .^ 2,
        terminal, Matrix(Diagonal(scale[1:6] ./ terminal)))
end

"SI covariance D P~ D of a normalized covariance P~."
psyche_si_covariance(u::PsycheUncertainty, P) = u.factor_si * P * u.factor_si'

"Normalized-to-node-units transform diag(1 / node_scales) D of node k."
psyche_node_transform(d::PsycheData, u::PsycheUncertainty, k) = Diagonal(1 ./ d.node_scales[:, k]) * u.factor_si

"UT parameters of a phase: its units and normalization, thrust (none on Mars), exhaust velocity and acceleration diffusion."
function psyche_ut_params(d::PsycheData, o::Options, u::PsycheUncertainty, phase::Int)
    state, time = d.phase_scales[:, phase], d.time_scales[phase]
    tmax = phase == 2 ? 0.0 : d.thrust_max_n / (state[7] * state[1] / time^2)
    return ut_params(o; scale = u.covariance_scale ./ state, nav_cov = u.navigation_covariance, tmax,
        gravity = HeliocentricGravity(), ve = d.isp_s * 9.80665 / state[4],
        sigma_nd = o.acceleration_diffusion_km_s32 * 1000 * time^1.5 / state[1])
end

psyche_phase_params(d::PsycheData, o::Options, u::PsycheUncertainty) = [psyche_ut_params(d, o, u, phase) for phase in 1:3]

"""
End state of arc k in its end node's units from theta = (start mean 7, feedforward 3,
Mars days 2): one XMDS2 RK9 step of the nominal dynamics in the phase's units (mass flow
of the executed command, none on Mars) and, on a switch arc, the change of origin at the
SOI epoch with the stored ephemeris. Generic in the number type of theta.
"""
function psyche_arc_end(d::PsycheData, params, k::Int, theta)
    p = params[d.phase[k]]
    u = p.tmax * SVector{3}(theta[8], theta[9], theta[10])
    return psyche_arc_step(d, params, k, SVector{7}(ntuple(i -> theta[i], Val(7))), u, sigma_mass_flow(p, u),
        (theta[11], theta[12]))
end

"""
The same from theta = (start mean 7, direction 3, throttle, Mars days 2), the deterministic optimisation's
controls: thrust T_max Gamma d and mass flow T_max Gamma (no smoothing).
"""
function psyche_arc_end_throttle(d::PsycheData, params, k::Int, theta)
    p = params[d.phase[k]]
    u = p.tmax * theta[11] * SVector{3}(theta[8], theta[9], theta[10])
    return psyche_arc_step(d, params, k, SVector{7}(ntuple(i -> theta[i], Val(7))), u, p.tmax * theta[11],
        (theta[12], theta[13]))
end

"One RK9 step of arc k from mean m with thrust u and mass-flow magnitude `flow`, then the origin change of a switch arc."
function psyche_arc_step(d::PsycheData, params, k::Int, m, u, flow, days)
    phase = d.phase[k]
    p = params[phase]
    h = arc_duration(d, k, days) / d.time_scales[phase]
    y = rk_step(XMDS2RK9, m, u, flow, zero(SVector{7,Float64}), h, p.gravity, p.ve)
    physical = y .* SVector{7}(ntuple(i -> d.phase_scales[i, phase], Val(7)))
    if d.switch[k]
        et = phase == 1 ? d.epochs[2] : mars_exit_epoch(d, days)
        planet = d.planet(et)
        sign = phase == 1 ? -1 : 1
        physical = physical + sign * SVector{7}(ntuple(i -> i <= 6 ? planet[i] : zero(planet[1]), Val(7)))
    end
    return physical ./ SVector{7}(ntuple(i -> d.node_scales[i, k+1], Val(7)))
end

"Origin translation of a switch arc's end node at fixed Mars days, in the end node's units (zero otherwise)."
function switch_translation(d::PsycheData, k::Int, days)
    d.switch[k] || return zeros(NX)
    et = d.phase[k] == 1 ? d.epochs[2] : mars_exit_epoch(d, days)
    sign = d.phase[k] == 1 ? -1 : 1
    return [sign .* collect(d.planet(et)); 0.0] ./ d.node_scales[:, k+1]
end

# Mars encounter geometry at the nominal closest approach (moving, focusing-scaled axes): the
# osculating hyperbolic B-plane with moving S/T/R axes, its gravitational-focusing scaling,
# the analytic Jacobian of the scaled coordinates, and the event map to each dispersed
# trajectory's own closest approach. All functions are generic in the number type, so the
# clearance oracle differentiates them exactly with dual numbers.

"Osculating hyperbola, incoming asymptote and B-plane axes of the Mars-relative state z = (r, v) [SI]."
@inline function bplane_basis(z, mu, rho)
    r, v = SVector(z[1], z[2], z[3]), SVector(z[4], z[5], z[6])
    rn = norm(r)
    c = dot(v, v) - 2mu / rn                      # C3
    h = cross(r, v)
    hn = norm(h)
    ev = cross(v, h) / mu - r / rn
    e = norm(ev)
    p = ev / e
    q = cross(h, p) / hn
    w = sqrt(e * e - 1)
    s = (p + w * q) / e                            # incoming asymptote
    pole = SVector(0.0, 0.0, -1.0)
    vt = cross(pole, s)
    tn = norm(vt)
    t = vt / tn
    axis_r = cross(s, t)
    speed = sqrt(c)
    br, bt = dot(t, h) / speed, -dot(axis_r, h) / speed
    focus = sqrt(1 + 2mu / (rho * c))
    return (; r, v, rn, c, h, hn, ev, e, p, q, w, s, pole, tn, t, axis_r, speed, br, bt, focus)
end

"Scaled B-plane coordinates b = (B_R, B_T) / lambda_rho [m]."
@inline function scaled_bplane(z, mu, rho)
    b = bplane_basis(z, mu, rho)
    return SVector(b.br, b.bt) / b.focus
end

"Analytic 2 x 6 Jacobian of the scaled B-plane coordinates, moving axes and focusing included."
@inline function bplane_jacobian(z, mu, rho)
    b = bplane_basis(z, mu, rho)
    T = eltype(b.r)
    columns = ntuple(Val(6)) do k
        dr = SVector{3,T}(ntuple(i -> T(i == k), Val(3)))
        dv = SVector{3,T}(ntuple(i -> T(i + 3 == k), Val(3)))
        dc = 2dot(b.v, dv) + 2mu * dot(b.r, dr) / b.rn^3
        dspeed = dc / (2b.speed)
        dh = cross(dr, b.v) + cross(b.r, dv)
        dhn = dot(b.h, dh) / b.hn
        dev = (cross(dv, b.h) + cross(b.v, dh)) / mu - dr / b.rn + b.r * dot(b.r, dr) / b.rn^3
        de = dot(b.ev, dev) / b.e
        dp = dev / b.e - b.ev * de / b.e^2
        dq = (cross(dh, b.p) + cross(b.h, dp) - dhn * b.q) / b.hn
        ds = -de * b.p / b.e^2 + dp / b.e + de * b.q / (b.e^2 * b.w) + b.w * dq / b.e
        dt0 = cross(b.pole, ds)
        dt = (dt0 - dot(b.t, dt0) * b.t) / b.tn
        dar = cross(ds, b.t) + cross(b.s, dt)
        dbr = (dot(b.t, dh) + dot(b.h, dt) - b.br * dspeed) / b.speed
        dbt = -(dot(b.axis_r, dh) + dot(b.h, dar) + b.bt * dspeed) / b.speed
        dfocus = -2mu * dspeed / (b.focus * rho * b.speed^3)
        (SVector(dbr, dbt) - SVector(b.br, b.bt) * dfocus / b.focus) / b.focus
    end
    return hcat(columns...)
end

"""
Clearance of the closest approach z (Mars-relative, SI) with position/velocity covariance
P and nominal acceleration `acc`, in units of rho:
||b|| / rho - 1 - chi sqrt(lambda_max(C_b)) / rho >= 0, with C_b = J E P E' J' and the event
map E = I - [v; a][v; r]' / q to each trajectory's own closest approach (q = v.v + r.a).
The largest eigenvalue is unsmoothed.
"""
@inline function encounter_clearance(z, P, acc, mu, rho, chi)
    r, v = SVector(z[1], z[2], z[3]), SVector(z[4], z[5], z[6])
    q = dot(v, v) + dot(r, acc)
    event_map = SMatrix{6,6}(1.0I) - vcat(v, acc) * vcat(v, r)' / q
    J = bplane_jacobian(z, mu, rho) * event_map
    C = J * P * J' / rho^2
    a, c, d = C[1, 1], C[1, 2], C[2, 2]
    largest = (a + d) / 2 + sqrt(((a - d) / 2)^2 + c^2)
    return norm(scaled_bplane(z, mu, rho)) / rho - 1 - chi * sqrt(largest)
end

"Encounter statistics of a mean state and covariance at the closest approach (SI), for the report."
function encounter_report(z, P, mu, rho, chi)
    r = SVector(z[1], z[2], z[3])
    acc = -mu * r / norm(r)^3
    v = SVector(z[4], z[5], z[6])
    q = dot(v, v) + dot(r, acc)
    E = SMatrix{6,6}(1.0I) - vcat(v, acc) * vcat(v, r)' / q
    J = bplane_jacobian(z, mu, rho) * E
    timing = -vcat(v, r)' / q
    joint = vcat(J, timing) * P * vcat(J, timing)'
    b = scaled_bplane(z, mu, rho)
    return Dict{String,Any}("b_scaled_m" => collect(b), "b_covariance_m2" => [collect(row) for row in eachrow(Matrix(joint[1:2, 1:2]))],
        "closest_approach_time_std_s" => sqrt(joint[3, 3]), "clearance" => encounter_clearance(z, P, acc, mu, rho, chi),
        "closest_approach_altitude_m" => norm(r) - (rho - PSYCHE_MINIMUM_MARS_ALTITUDE_M),
        "clearance_radius_m" => rho, "chi" => chi)
end
