"""Psyche input geometry, event-following mesh and stochastic arc maps."""

from integrator.stochastic import nominal_map as nominal_map
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from pathlib import Path
import json
from scipy.interpolate import PPoly
from scipy.optimize import brentq
from deterministic_trajectory_optimization.psyche import (
    Mission,
    Settings,
    load_solution,
    evaluate_phase,
)
from types import SimpleNamespace
import stochastic_trajectory_optimization.dynamics as common


def launch_covariance_si(position_std_m=2e6):
    c, alpha, delta = 34.0, 1.9845214, 0.4888475
    s = np.array(
        [np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha), np.sin(delta)]
    )
    da = np.array([-np.cos(delta) * np.sin(alpha), np.cos(delta) * np.cos(alpha), 0.0])
    dd = np.array(
        [-np.sin(delta) * np.cos(alpha), -np.sin(delta) * np.sin(alpha), np.cos(delta)]
    )
    jac = np.column_stack((s / (2 * np.sqrt(c)), np.sqrt(c) * da, np.sqrt(c) * dd))
    pq = np.diag((np.array([0.25, 0.0087266, 0.0043633]) / 3) ** 2)
    result = np.zeros((7, 7))
    result[:3, :3] = np.eye(3) * position_std_m**2
    result[3:6, 3:6] = 1e6 * jac @ pq @ jac.T  # km/s -> m/s, including cross terms
    result[6, 6] = 1e-6
    return result


def bplane_vector(z, mu):
    """Unscaled osculating B-plane coordinates [B.R, B.T] of a planet-relative state."""
    r, v = z[:3], z[3:6]
    radius = jnp.linalg.norm(r)
    c = v @ v - 2 * mu / radius
    h = jnp.cross(r, v)
    hn = jnp.linalg.norm(h)
    ev = jnp.cross(v, h) / mu - r / radius
    e = jnp.linalg.norm(ev)
    p = ev / e
    q = jnp.cross(h / hn, p)
    s = (p + jnp.sqrt(e * e - 1) * q) / e
    vt = jnp.cross(jnp.array([0.0, 0.0, -1.0]), s)
    tn = jnp.linalg.norm(vt)
    t = vt / tn
    axis_r = jnp.cross(s, t)
    b = jnp.array([t @ h, -axis_r @ h]) / jnp.sqrt(c)
    return b, jnp.array([c, e, hn, tn]), s


def bplane_geometry(z, mu, rho):
    b, domain, s = bplane_vector(z, mu)
    focusing = jnp.sqrt(1 + 2 * mu / (rho * domain[0]))
    return b / focusing, domain, s


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
    q = v @ v + r @ acceleration_si
    time_gradient = -normal / q
    event_map = jnp.eye(6) + jnp.outer(f, time_gradient)
    b, domain, s = bplane_geometry(z, mu, rho)
    jacobian = jax.jacfwd(scaled_bplane)(z, mu, rho)
    joint_map = jnp.vstack((jacobian @ event_map, time_gradient))
    joint_covariance = joint_map @ covariance_si @ joint_map.T
    cb = joint_covariance[:2, :2] / rho**2
    a, cross, d = cb[0, 0], cb[0, 1], cb[1, 1]
    upper = 0.5 * (a + d) + jnp.sqrt((0.5 * (a - d)) ** 2 + cross**2)
    risk_multiplier = jnp.sqrt(-2 * jnp.log(0.001))
    clearance = jnp.linalg.norm(b) / rho - 1 - risk_multiplier * jnp.sqrt(upper)
    return dict(
        b_si=b,
        b_covariance_si=joint_covariance[:2, :2],
        joint_b_time_covariance=joint_covariance,
        time_variance_s2=joint_covariance[2, 2],
        event_map=event_map,
        jacobian=jacobian,
        q=q,
        domain=domain,
        s_axis=s,
        chance_clearance=clearance,
        upper_covariance_eigenvalue=upper,
    )


def analytic_bplane_jacobian(z, mu, rho):
    """Independent NumPy directional derivatives of the moving B-plane map."""
    r, v = np.asarray(z[:3]), np.asarray(z[3:6])
    rn = np.linalg.norm(r)
    c = v @ v - 2 * mu / rn
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    ev = np.cross(v, h) / mu - r / rn
    e = np.linalg.norm(ev)
    if c <= 0 or e <= 1 or hn <= 0:
        raise ValueError("B-plane requires a nonradial hyperbolic state")
    p = ev / e
    q = np.cross(h, p) / hn
    w = np.sqrt(e * e - 1)
    s = (p + w * q) / e
    pole = np.array([0.0, 0.0, -1.0])
    tn = np.linalg.norm(np.cross(pole, s))
    if tn <= 1e-12:
        raise ValueError("B-plane reference axis is singular")
    t = np.cross(pole, s) / tn
    axis_r = np.cross(s, t)
    speed = np.sqrt(c)
    br, bt = (t @ h) / speed, -(axis_r @ h) / speed
    focus = np.sqrt(1 + 2 * mu / (rho * c))
    jac = np.empty((2, 6))
    for k in range(6):
        dr, dv = np.eye(6)[k, :3], np.eye(6)[k, 3:]
        dc = 2 * v @ dv + 2 * mu * (r @ dr) / rn**3
        dspeed = dc / (2 * speed)
        dh = np.cross(dr, v) + np.cross(r, dv)
        dhn = (h @ dh) / hn
        dev = (np.cross(dv, h) + np.cross(v, dh)) / mu - dr / rn + r * (r @ dr) / rn**3
        de = (ev @ dev) / e
        dp = dev / e - ev * de / e**2
        dq = (np.cross(dh, p) + np.cross(h, dp) - dhn * q) / hn
        ds = -de * p / e**2 + dp / e + de * q / (e**2 * w) + w * dq / e
        cross = np.cross(pole, ds)
        dt = (cross - (t @ cross) * t) / tn
        dar = np.cross(ds, t) + np.cross(s, dt)
        dbr = (t @ dh + h @ dt - br * dspeed) / speed
        dbt = -(axis_r @ dh + h @ dar + bt * dspeed) / speed
        dfocus = -2 * mu * dspeed / (focus * rho * speed**3)
        jac[:, k] = (np.array([dbr, dbt]) - np.array([br, bt]) * dfocus / focus) / focus
    return jac


ROOT = Path(__file__).resolve().parents[2]
DAY = 86400.0
# Numerical intervals per interval of the deterministic reference mesh (states from the
# reference's own shooting propagation, the interval's thrust on every piece). An integer
# r splits every interval into r equal pieces; 1 < r < 2 halves the round((r-1) n) longest
# intervals of each phase (ties: the earlier interval), so the mesh grows by the factor r.
MESH_REFINEMENT = 1.5


def refined_mesh(mesh, refinement):
    """Phase mesh (fractions in [0, 1]) refined by `refinement`; every original node is kept."""
    if float(refinement).is_integer():
        pieces = np.arange(int(refinement)) / int(refinement)
        return np.r_[
            (mesh[:-1, None] + np.diff(mesh)[:, None] * pieces).ravel(), mesh[-1]
        ]
    widths = np.diff(mesh)
    count = int(np.floor((refinement - 1) * len(widths) + 0.5))
    split = np.sort(np.argsort(-widths, kind="stable")[:count])
    return np.sort(np.r_[mesh, (mesh[split] + mesh[split + 1]) / 2])


@dataclass
class Arc:
    """One numerical interval from node `start` to node `end` (= start+1)."""

    phase: int
    start: int
    end: int
    control: int
    # duration [s] = constant + coefficients @ [inbound_days, outbound_days]
    duration: np.ndarray
    # The last arc of a phase ends with an origin translation.
    switch: bool = False


def polynomial_ephemeris(cache, body):
    """Convert the actual six stored cubic splines, preserving their derivatives."""
    spline = cache._splines[body]
    polys = [PPoly.from_spline((spline.t, spline.c[:, j], spline.k)) for j in range(6)]
    knots = jnp.asarray(polys[0].x)
    coefficients = jnp.asarray(np.stack([p.c for p in polys], axis=-1))
    origin = cache.origin_et
    lo, hi = cache.valid_interval

    def state(et):
        day = (et - origin) / DAY
        index = jnp.clip(
            jnp.searchsorted(knots, day, side="right") - 1, 0, len(knots) - 2
        )
        offset = day - knots[index]
        c = coefficients[:, index, :]
        value = ((c[0] * offset + c[1]) * offset + c[2]) * offset + c[3]
        return jnp.where((et >= lo) & (et <= hi), value, jnp.full(6, jnp.nan))

    return state


class PsycheInput:
    """The converged reference mesh, refined, plus the closest-approach (CA) node.

    All original mesh nodes are retained; the mesh is refined by `mesh_refinement`
    (see `refined_mesh`), and every interval of the two solar cruises is then split into
    `cruise_subdivision` equal pieces (the Mars phase keeps its mesh). Mars is ballistic; the CA node splits
    an ordinary propagation interval (or reuses a coincident node) and
    introduces no command or reset.
    """

    def __init__(
        self,
        path=None,
        mesh_refinement=MESH_REFINEMENT,
        initial_position_std_m=2e6,
        terminal_std_si=(2e6, 2.0),
        cruise_subdivision=1,
    ):
        if not (
            mesh_refinement >= 1
            and (float(mesh_refinement).is_integer() or mesh_refinement < 2)
        ):
            raise ValueError(
                "mesh_refinement must be a positive integer or lie between 1 and 2"
            )
        if not (float(cruise_subdivision).is_integer() and cruise_subdivision >= 1):
            raise ValueError("cruise_subdivision must be a positive integer")
        self.mesh_refinement = float(mesh_refinement)
        self.cruise_subdivision = int(cruise_subdivision)
        self.path = Path(
            path
            or ROOT
            / "output/psyche_mission_deterministic_traj_opt/energy_optimal/psyche.npz"
        )
        solution = self.solution = load_solution(self.path)
        if (
            solution["objective"] not in ("energy", "fuel")
            or not solution["solver_success"]
        ):
            raise ValueError(
                "Psyche requires a converged deterministic energy or fuel reference"
            )
        with np.load(self.path, allow_pickle=False) as archive:
            defects = (
                archive["interval_defect_scaled"]
                if "interval_defect_scaled" in archive
                else None
            )
        if defects is None:
            companion = self.path.with_name(self.path.stem + "_defects.json")
            if not companion.exists():
                raise ValueError(
                    "The deterministic reference needs saved local-defect evidence"
                )
            evidence = json.loads(companion.read_text(encoding="utf-8"))
            defects = np.concatenate(
                [p["relative"] for p in evidence["interval_defects"]]
            )
        if (
            not np.all(np.isfinite(defects))
            or np.max(defects) > solution["settings"]["defect_tolerance"]
        ):
            raise ValueError(
                "The deterministic reference did not meet its local-defect tolerance"
            )
        if len(solution["phases"]) != 3:
            raise ValueError(
                "Regenerate deterministic Psyche from fixed Earth SOI exit first"
            )
        self.mission = mission = Mission(Settings.from_dict(solution["settings"]))
        if not mission.settings.ballistic_mars or np.any(
            solution["phases"][1]["thrust_n"] != 0
        ):
            raise ValueError(
                "Reoptimise deterministic Psyche with ballistic Mars first"
            )
        self.planet = polynomial_ephemeris(mission.cache, 499)
        self.initial_covariance_si = launch_covariance_si(initial_position_std_m)
        self.covariance_scale = np.sqrt(np.diag(self.initial_covariance_si))
        self.initial_covariance = self.initial_covariance_si / np.outer(
            self.covariance_scale, self.covariance_scale
        )
        self.navigation_covariance = np.diag(
            (np.r_[[2e5] * 3, [0.1] * 3] / self.covariance_scale[:6]) ** 2
        )
        self.terminal_inverse_std = (
            self.covariance_scale[:6]
            / np.r_[[terminal_std_si[0]] * 3, [terminal_std_si[1]] * 3]
        )
        mars = solution["phases"][1]

        def radial(tau):
            y = evaluate_phase(solution, 1, tau, mission)
            return float(y[:3] @ y[3:6])

        # Search continuous arcs, including subdivisions, rather than select a node.
        samples = np.unique(np.r_[mars["mesh"], np.linspace(0, 1, 4001)])
        values = np.array([radial(t) for t in samples])
        candidates = []
        for k in np.flatnonzero((values[:-1] <= 0) & (values[1:] > 0)):
            tau = brentq(radial, samples[k], samples[k + 1], xtol=1e-14)
            state = evaluate_phase(solution, 1, tau, mission)
            candidates.append((np.linalg.norm(state[:3]), tau))
        if not candidates:
            raise ValueError("No interior continuous Mars closest approach found")
        _, ca = min(candidates)
        if (
            min(np.linalg.norm(mars["states_si"][i, :3]) for i in (0, -1))
            <= min(candidates)[0]
        ):
            raise ValueError(
                "Mars minimum is an endpoint, not the selected interior encounter"
            )
        self.reference_ca_fraction = ca
        # Reuse an existing node if the stationary point already lies on it.
        nearest = int(np.argmin(abs(mars["mesh"] - ca)))
        if abs(mars["mesh"][nearest] - ca) < 1e-12:
            ca = float(mars["mesh"][nearest])
        self.seed_durations = np.array([ca, 1 - ca]) * mars["duration_s"] / DAY
        self.arcs, self.node_origins, states, controls = [], [], [], []
        for phase, record in enumerate(solution["phases"]):
            fine = refined_mesh(record["mesh"], self.mesh_refinement)
            if phase != 1 and self.cruise_subdivision > 1:
                fine = refined_mesh(fine, self.cruise_subdivision)
            mesh = np.unique(np.r_[fine, ca]) if phase == 1 else fine
            if phase == 0:
                self.node_origins.append(0)
                states.append(record["states_si"][0])
            for k, (left, right) in enumerate(zip(mesh[:-1], mesh[1:])):
                start = len(states) - 1
                if phase == 0:
                    duration = np.array(
                        [(mission.tm - mission.t0) * (right - left), 0.0, 0.0]
                    )
                elif phase == 2:
                    duration = (right - left) * np.array(
                        [mission.tf - mission.tm, -DAY, -DAY]
                    )
                elif right <= ca:
                    duration = np.array([0.0, DAY * (right - left) / ca, 0.0])
                else:
                    duration = np.array([0.0, 0.0, DAY * (right - left) / (1 - ca)])
                switch = k == len(mesh) - 2 and phase < 2
                state = (
                    solution["phases"][phase + 1]["states_si"][0]
                    if switch
                    else evaluate_phase(solution, phase, right, mission)
                )
                self.node_origins.append(phase + 1 if switch else phase)
                states.append(state)
                old = min(
                    np.searchsorted(record["mesh"], (left + right) / 2, side="right")
                    - 1,
                    len(record["mesh"]) - 2,
                )
                controls.append(
                    np.zeros(3)
                    if phase == 1
                    else record["thrust_n"][old] / mission.settings.thrust_max_n
                )
                self.arcs.append(
                    Arc(
                        phase,
                        start,
                        len(states) - 1,
                        len(controls) - 1,
                        duration,
                        switch=switch,
                    )
                )
                if phase == 1 and right == ca:
                    self.ca_node = len(states) - 1
                if switch and phase == 0:
                    self.mars_entry_node = len(states) - 1
                if switch and phase == 1:
                    self.mars_exit_node = len(states) - 1
                    self.mars_exit_arc = len(self.arcs) - 1
        self.node_scales = np.array(
            [mission.scales[i].state for i in self.node_origins]
        )
        self.seed_states = np.array(states) / self.node_scales
        self.seed_controls = np.array(controls)
        # Arc-indexed arrays include Mars arcs with zero control. Only powered
        # (cruise) slots are decision variables in either stochastic transcription.
        self.n_controls = len(controls)
        self.powered_arcs = np.array(
            [k for k, a in enumerate(self.arcs) if a.phase != 1], dtype=int
        )
        self.control_indices = np.full(self.n_controls, -1, dtype=int)
        self.control_indices[self.powered_arcs] = np.arange(len(self.powered_arcs))
        self.n_powered_controls = len(self.powered_arcs)
        self.n_intervals = len(states) - 1
        self.n_nodes = len(states)
        self.phase_mesh_counts = [
            sum(a.phase == p for a in self.arcs) for p in range(3)
        ]
        assert self.n_controls == self.n_intervals

    def duration(self, arc, days):
        return arc.duration[0] + jnp.asarray(arc.duration[1:]) @ days

    def transform_end(self, arc, value, days):
        if not arc.switch:
            return value
        mission = self.mission
        scale = jnp.asarray(mission.scales[arc.phase].state)
        target_scale = jnp.asarray(mission.scales[arc.phase + 1].state)
        et = mission.tm if arc.phase == 0 else mission.tm + DAY * jnp.sum(days)
        translation = jnp.concatenate((self.planet(et), jnp.zeros(1)))
        return (
            value * scale + (-1 if arc.phase == 0 else 1) * translation
        ) / target_scale


def phase_case(mission, phase):
    scale, settings = mission.scales[phase], mission.settings
    force = scale.mass * scale.length / scale.time**2
    return SimpleNamespace(
        dynamics_model="heliocentric",
        test_case_id="psyche",
        time_unit=scale.time,
        length_unit=scale.length / 1000,
        velocity_unit=scale.velocity / 1000,
        max_thrust_nd=0.0
        if mission.is_ballistic(phase)
        else settings.thrust_max_n / force,
        exhaust_velocity_nd=settings.isp_s * 9.80665 / scale.velocity,
        m0_wet=settings.wet_mass_kg,
        thrust_unit=force,
    )


class Maps:
    def __init__(self, data, options, integration_refinement=1):
        self.data, self.options = data, options
        self.nominal, self.moments, self.cases, self.normalizations = [], [], [], []
        self.nominal_samples = []
        for i in range(3):
            case = phase_case(data.mission, i)
            norm = common.Normalization(
                data.covariance_scale / data.mission.scales[i].state,
                data.covariance_scale / data.mission.scales[i].state,
            )
            self.cases.append(case)
            self.normalizations.append(norm)
            self.nominal.append(
                nominal_map(case, options.integrator_substeps, options.control_norm_eps)
            )
            self.nominal_samples.append(
                nominal_map(
                    case,
                    options.integrator_substeps,
                    options.control_norm_eps,
                    save_substeps=True,
                )
            )
            self.moments.append(
                common.build_propagation_arc_map(
                    case, options, norm, integration_refinement
                )
            )

    def deterministic(self, arc, mean, feed, days):
        """Nominal end state of one interval, in the next node's origin and scale."""
        duration = self.data.duration(arc, days) / self.cases[arc.phase].time_unit
        end = self.nominal[arc.phase](mean, feed, duration)
        return self.data.transform_end(arc, end, days)

    def deterministic_surface_samples(self, arc, mean, feed, days):
        """Clearance at the existing uniform RK9 endpoints, in the arc's origin."""
        mission = self.data.mission
        if arc.phase != 1:
            return jnp.zeros(0)
        radius = (mission.settings.mars_radius_m + 500e3) / mission.scales[
            arc.phase
        ].length
        duration = self.data.duration(arc, days) / self.cases[arc.phase].time_unit
        samples = self.nominal_samples[arc.phase](mean, feed, duration)
        return jnp.sum(samples[:, :3] ** 2, axis=1) - radius**2

    def stochastic(self, arc, mean, covariance, feed, gain, days):
        """Mean, covariance and executed-control covariance after one interval.

        Mars arcs propagate with zero command and gain, so their executed-control
        covariance is exactly zero.
        """
        duration = self.data.duration(arc, days) / self.cases[arc.phase].time_unit
        if arc.phase == 1:
            feed, gain = jnp.zeros(3), jnp.zeros((3, 6))
        end, p, executed_cov = self.moments[arc.phase](
            mean, covariance, feed, gain, duration
        )
        if arc.phase == 1:
            executed_cov = jnp.zeros((3, 3))
        return self.data.transform_end(arc, end, days), p, executed_cov
