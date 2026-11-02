"""SI launch covariance and moving-axis, event-aware scaled B-plane geometry.

Equations follow Sections 5 and 9--12 of the user-supplied 2026-10-25 proposal.
Hyperbolic validity is checked explicitly; no elliptic-to-hyperbolic clipping.
"""
import numpy as np
import jax
import jax.numpy as jnp


def launch_covariance_si(position_std_m=2e6):
    c, alpha, delta = 34., 1.9845214, .4888475
    s = np.array([np.cos(delta)*np.cos(alpha), np.cos(delta)*np.sin(alpha), np.sin(delta)])
    da = np.array([-np.cos(delta)*np.sin(alpha), np.cos(delta)*np.cos(alpha), 0.])
    dd = np.array([-np.sin(delta)*np.cos(alpha), -np.sin(delta)*np.sin(alpha), np.cos(delta)])
    jac = np.column_stack((s/(2*np.sqrt(c)), np.sqrt(c)*da, np.sqrt(c)*dd))
    pq = np.diag((np.array([.25, .0087266, .0043633])/3)**2)
    result = np.zeros((7, 7))
    result[:3, :3] = np.eye(3)*position_std_m**2
    result[3:6, 3:6] = 1e6*jac@pq@jac.T  # km/s -> m/s, including cross terms
    result[6, 6] = 1e-6
    return result


def bplane_vector(z, mu):
    """Unscaled osculating B-plane coordinates [B.R, B.T] of a planet-relative state."""
    r, v = z[:3], z[3:6]
    radius = jnp.linalg.norm(r)
    c = v@v - 2*mu/radius
    h = jnp.cross(r, v)
    hn = jnp.linalg.norm(h)
    ev = jnp.cross(v, h)/mu - r/radius
    e = jnp.linalg.norm(ev)
    p = ev/e
    q = jnp.cross(h/hn, p)
    s = (p+jnp.sqrt(e*e-1)*q)/e
    vt = jnp.cross(jnp.array([0., 0., -1.]), s)
    tn = jnp.linalg.norm(vt)
    t = vt/tn
    axis_r = jnp.cross(s, t)
    b = jnp.array([t@h, -axis_r@h])/jnp.sqrt(c)
    return b, jnp.array([c, e, hn, tn]), s


def bplane_geometry(z, mu, rho):
    b, domain, s = bplane_vector(z, mu)
    focusing = jnp.sqrt(1+2*mu/(rho*domain[0]))
    return b/focusing, domain, s


def scaled_bplane(z, mu, rho):
    return bplane_geometry(z, mu, rho)[0]


def encounter_statistics(z, covariance_si, acceleration_si, mu, rho):
    """Covariance at each trajectory's CA from covariance at the common epoch.

    Both moving B-plane axes and the focusing factor are differentiated. The
    event-time and B-plane correlations are returned, without double counting
    an independent time variance. The largest B-plane covariance eigenvalue is
    unsmoothed: repeated eigenvalues are nondifferentiable and no artificial
    floor is substituted.
    """
    r, v = z[:3], z[3:6]
    f = jnp.concatenate((v, acceleration_si))
    normal = jnp.concatenate((v, r))
    q = v@v + r@acceleration_si
    time_gradient = -normal/q
    event_map = jnp.eye(6) + jnp.outer(f, time_gradient)
    b, domain, s = bplane_geometry(z, mu, rho)
    jacobian = jax.jacfwd(scaled_bplane)(z, mu, rho)
    joint_map = jnp.vstack((jacobian@event_map, time_gradient))
    joint_covariance = joint_map@covariance_si@joint_map.T
    cb = joint_covariance[:2, :2]/rho**2
    a, cross, d = cb[0, 0], cb[0, 1], cb[1, 1]
    upper = .5*(a+d) + jnp.sqrt((.5*(a-d))**2+cross**2)
    risk_multiplier = jnp.sqrt(-2*jnp.log(.001))
    clearance = jnp.linalg.norm(b)/rho - 1 - risk_multiplier*jnp.sqrt(upper)
    return dict(b_si=b, b_covariance_si=joint_covariance[:2, :2],
                joint_b_time_covariance=joint_covariance, time_variance_s2=joint_covariance[2, 2],
                event_map=event_map, jacobian=jacobian, q=q, domain=domain, s_axis=s,
                chance_clearance=clearance, upper_covariance_eigenvalue=upper)


def analytic_bplane_jacobian(z, mu, rho):
    """Independent NumPy directional derivatives from proposal Eqs. 10.1--10.9."""
    r, v = np.asarray(z[:3]), np.asarray(z[3:6])
    rn = np.linalg.norm(r)
    c = v@v-2*mu/rn
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    ev = np.cross(v, h)/mu-r/rn
    e = np.linalg.norm(ev)
    if c <= 0 or e <= 1 or hn <= 0:
        raise ValueError('B-plane requires a nonradial hyperbolic state')
    p = ev/e
    q = np.cross(h, p)/hn
    w = np.sqrt(e*e-1)
    s = (p+w*q)/e
    pole = np.array([0., 0., -1.])
    tn = np.linalg.norm(np.cross(pole, s))
    if tn <= 1e-12:
        raise ValueError('B-plane reference axis is singular')
    t = np.cross(pole, s)/tn
    axis_r = np.cross(s, t)
    speed = np.sqrt(c)
    br, bt = (t@h)/speed, -(axis_r@h)/speed
    focus = np.sqrt(1+2*mu/(rho*c))
    jac = np.empty((2, 6))
    for k in range(6):
        dr, dv = np.eye(6)[k, :3], np.eye(6)[k, 3:]
        dc = 2*v@dv+2*mu*(r@dr)/rn**3
        dspeed = dc/(2*speed)
        dh = np.cross(dr, v)+np.cross(r, dv)
        dhn = (h@dh)/hn
        dev = (np.cross(dv, h)+np.cross(v, dh))/mu-dr/rn+r*(r@dr)/rn**3
        de = (ev@dev)/e
        dp = dev/e-ev*de/e**2
        dq = (np.cross(dh, p)+np.cross(h, dp)-dhn*q)/hn
        ds = -de*p/e**2+dp/e+de*q/(e**2*w)+w*dq/e
        cross = np.cross(pole, ds)
        dt = (cross-(t@cross)*t)/tn
        dar = np.cross(ds, t)+np.cross(s, dt)
        dbr = (t@dh+h@dt-br*dspeed)/speed
        dbt = -(axis_r@dh+h@dar+bt*dspeed)/speed
        dfocus = -2*mu*dspeed/(focus*rho*speed**3)
        jac[:, k] = (np.array([dbr, dbt])-np.array([br, bt])*dfocus/focus)/focus
    return jac
