# Mars encounter geometry at the nominal closest approach (Farnocchia et al. 2019): the
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
