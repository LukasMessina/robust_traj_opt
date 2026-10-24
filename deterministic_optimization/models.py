"""Physical cases and dynamics shared with stochastic consumers."""

from __future__ import annotations
from dataclasses import dataclass, field
import casadi
import numpy as np
import integrator


@dataclass(frozen=True)
class SpacecraftScaling:
    m0_wet: float = 1000.0
    m0_dry: float = 500.0
    thrust_max: float = 0.5
    isp: float = 2000.0

    @property
    def velocity_unit(self) -> float:
        return self.length_unit / self.time_unit  # [km/s]

    @property
    def thrust_unit(self) -> float:
        return self.m0_wet * 1000.0 * self.velocity_unit / self.time_unit  # [N]

    @property
    def m0_wet_nd(self) -> float:
        return self.m0_wet / self.m0_wet  # [-]

    @property
    def m0_dry_nd(self) -> float:
        return self.m0_dry / self.m0_wet  # [-]

    @property
    def max_thrust_nd(self) -> float:
        return self.thrust_max / self.thrust_unit  # [-]

    @property
    def max_thrust_n(self) -> float:
        return self.thrust_max  # [N]

    @property
    def exhaust_velocity_nd(self) -> float:
        return 9.80665 * self.isp / (1000.0 * self.velocity_unit)  # [-]


@dataclass(frozen=True)
class CR3BPEarthMoon(SpacecraftScaling):
    gm1: float = 398600.0  # km^3/s^2
    gm2: float = 4902.80
    length_unit: float = 384399.0  # km
    moon_radius: float = 1737.5  # km

    @property
    def mu(self) -> float:
        return self.gm2 / (self.gm1 + self.gm2)

    @property
    def time_unit(self) -> float:
        return np.sqrt(self.length_unit**3 / (self.gm1 + self.gm2))

    @property
    def moon_radius_lu(self) -> float:
        return self.moon_radius / self.length_unit


@dataclass(frozen=True)
class TestCase(SpacecraftScaling):
    test_case_id: str = ""
    display_name: str = ""
    tof_days: float = 0.0  # [days]
    nodes: int = 0

    x0: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # [-]
    xf: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # [-]

    departure_label: str = "departure orbit"
    target_label: str = "target orbit"
    departure_period: float | None = None  # [-]
    target_period: float | None = None  # [-]
    departure_period_days: float | None = None  # [days]
    target_period_days: float | None = None  # [days]

    @property
    def tof_nd(self) -> float:
        return self.tof_days * 86400.0 / self.time_unit  # [-]

    @property
    def x0_state(self) -> np.ndarray:
        return np.asarray(self.x0, dtype=float)  # [-]

    @property
    def xf_state(self) -> np.ndarray:
        return np.asarray(self.xf, dtype=float)  # [-]

    @property
    def x0_augmented_state(self) -> np.ndarray:
        return np.r_[self.x0_state, self.m0_wet_nd]  # [-]

    @property
    def xf_augmented_state(self) -> np.ndarray:
        return np.r_[self.xf_state, self.m0_wet_nd]  # [-]

    @property
    def departure_period_nd(self) -> float | None:
        if self.departure_period is not None:
            return self.departure_period  # [-]
        if self.departure_period_days is not None:
            return self.departure_period_days * 86400.0 / self.time_unit
        return None

    @property
    def target_period_nd(self) -> float | None:
        if self.target_period is not None:
            return self.target_period  # [-]
        if self.target_period_days is not None:
            return self.target_period_days * 86400.0 / self.time_unit
        return None


@dataclass(frozen=True)
class NrhoL2ToDro(TestCase, CR3BPEarthMoon):
    """L2 near-rectilinear halo orbit to distant retrograde orbit.

    Orbit periods are recovered here by minimising the closure error of a ballistic
    propagation from the tabulated state.
    """

    test_case_id: str = "nrho_l2_to_dro"
    display_name: str = "NRHO L2 to DRO"
    tof_days: float = 21.2  # [days]
    nodes: int = 300  # [-]
    x0: tuple[float, ...] = (
        1.02197,
        0.0,
        -0.18206,
        0.0,
        -0.10314,
        0.0,
    )  # [-]
    xf: tuple[float, ...] = (
        0.98337,
        0.25921,
        0.0,
        0.35134,
        -0.00833,
        0.0,
    )  # [-]
    departure_label: str = "NRHO L2"
    target_label: str = "DRO"
    departure_period: float = 1.5104038685  # [-]
    target_period: float = 3.3171701817  # [-]


@dataclass(frozen=True)
class LyapunovL1ToL2(TestCase, CR3BPEarthMoon):
    """Planar Lyapunov orbit at L1 to planar Lyapunov orbit at L2."""

    test_case_id: str = "lyapunov_l1_to_l2"
    display_name: str = "Lyapunov L1 to Lyapunov L2"
    tof_days: float = 12.0  # [days]
    nodes: int = 300  # [-]
    x0: tuple[float, ...] = (
        0.85599012364703531,
        0.12436459999999999,
        0.0,
        0.094844873498005022,
        0.044107030349277508,
        0.0,
    )  # [-]
    xf: tuple[float, ...] = (
        1.0959752057722425,
        0.11525999999999831,
        0.0,
        0.037470505824053729,
        0.12673805721118889,
        0.0,
    )  # [-]
    departure_label: str = "Lyapunov L1"
    target_label: str = "Lyapunov L2"
    departure_period: float = 2.9750964922007723  # [-]
    target_period: float = 3.49306635929003  # [-]


@dataclass(frozen=True)
class HaloL2ToHaloL1(TestCase, CR3BPEarthMoon):
    test_case_id: str = "halo_l2_to_halo_l1"
    display_name: str = "Halo L2 to Halo L1"  # [-]
    tof_days: float = 20.0  # [days]
    nodes: int = 300  # [-]
    x0: tuple[float, ...] = (
        1.1607973110000016,
        0.0,
        -0.12269696820337475,
        0.0,
        -0.20768326513738075,
        0.0,
    )  # [-]
    xf: tuple[float, ...] = (
        0.84871015300008812,
        0.0,
        0.17388998538319206,
        0.0,
        0.26350093896218163,
        0.0,
    )  # [-]
    departure_label: str = "Halo L2"
    target_label: str = "Halo L1"
    departure_period: float = 3.2746644337639852  # [-]
    target_period: float = 2.5748200748171399  # [-]


# Caleb (2025), Tabs. 3.2-3.3: Sun-centered inertial boundary states [km, km/s]; no epoch is given.
EARTH_MARS_TOF_DAYS = 348.79
EARTH_MARS_MU_SUN_KM3_S2 = 1.32712440041e11
EARTH_MARS_DEPARTURE_STATE_KM = (-140699693.0, -51614428.0, 980.0, 9.774596, -28.07828, 4.337725e-4)
EARTH_MARS_ARRIVAL_STATE_KM = (
    -172682023.0,
    176959469.0,
    7948912.0,
    -16.427384,
    -14.860506,
    9.21486e-2,
)


def earth_mars_data(samples: int = 1001) -> dict:
    """Boundary states and plotting tracks, in meters and meters/second.

    The tracks only serve the plots: two-body propagations of the boundary
    states covering at least one full period and the transfer duration.
    """
    from scipy.integrate import solve_ivp

    mu = EARTH_MARS_MU_SUN_KM3_S2

    def two_body(_, y):
        return np.r_[y[3:], -mu * y[:3] / np.linalg.norm(y[:3]) ** 3]

    def track(state, duration_s):
        times = np.linspace(0.0, duration_s, samples)
        sol = solve_ivp(
            two_body, (0.0, duration_s), state, method="DOP853", t_eval=times, rtol=1e-12, atol=1e-9
        )
        return times, sol.y * 1000.0

    def period(state):
        semi_major_axis = 1.0 / (2.0 / np.linalg.norm(state[:3]) - state[3:] @ state[3:] / mu)
        return 2.0 * np.pi * np.sqrt(semi_major_axis**3 / mu)

    earth0 = np.asarray(EARTH_MARS_DEPARTURE_STATE_KM)
    mars_f = np.asarray(EARTH_MARS_ARRIVAL_STATE_KM)
    tof_s = EARTH_MARS_TOF_DAYS * 86400.0
    earth_duration_s = max(period(earth0), tof_s)
    mars_duration_s = max(period(mars_f), tof_s)
    earth_times, earth = track(earth0, earth_duration_s)
    mars_times, mars = track(mars_f, -mars_duration_s)  # backward from arrival
    return {
        "source": "Caleb (2025), ISAE-SUPAERO PhD thesis, Sec. 3.2.2, Tabs. 3.2-3.3 (Lantoine and Russell)",
        "tof_days": EARTH_MARS_TOF_DAYS,
        "origin": "Sun",
        "frame": "Sun-centered inertial, as tabulated; no epoch given",
        "mu_sun_m3_s2": mu * 1e9,
        "earth_departure_state_si": (earth0 * 1000.0).tolist(),
        "mars_arrival_state_si": (mars_f * 1000.0).tolist(),
        "track_model": "two-body propagation of the boundary states (plots only)",
        "earth_track_times_days": (earth_times / 86400.0).tolist(),
        "mars_track_times_days": ((tof_s + mars_times[::-1]) / 86400.0).tolist(),
        "earth_track_duration_days": earth_duration_s / 86400.0,
        "earth_track_extent": "at least one two-body period from the departure state",
        "mars_track_duration_days": mars_duration_s / 86400.0,
        "mars_track_extent": "at least one two-body period ending at the arrival state",
        "earth_track_si": earth.tolist(),
        "mars_track_si": mars[:, ::-1].tolist(),
    }


# Fixed departure impulse added to Earth's velocity (a launch v-infinity), a boundary condition, not a
# decision variable. Direction: the primer vector at departure, i.e. the initial thrust direction of the
# impulse-free fuel-optimal transfer (Sun-centered J2000; 9.1 deg from Earth's velocity, tilted outward).
# To first order it removes the most cost per unit impulse; the energy-optimal one differs by ~3 deg.
# NOTE: this direction was computed for the Oguri--Lantoine case; recompute it for the Caleb case before
# using a nonzero magnitude.
DEPARTURE_DV_MAGNITUDE_KM_S = 0.0
DEPARTURE_DV_DIRECTION = (0.7659075403, 0.5819201041, 0.2734129336)


@dataclass(frozen=True)
class EarthMars(TestCase):
    """Caleb (2025), Sec. 3.2.2; Sun point mass, fixed boundary states, rendezvous."""

    dynamics_model = "heliocentric"
    test_case_id: str = "earth_mars"
    display_name: str = "Earth to Mars"
    tof_days: float = EARTH_MARS_TOF_DAYS
    nodes: int = 30  # retained stochastic model default; deterministic CLI uses 50
    m0_wet: float = 1000.0
    m0_dry: float = 500.0
    thrust_max: float = 0.5
    isp: float = 2000.0
    length_unit: float = 149597870.7  # km: exact astronomical unit
    departure_label: str = "Earth"
    target_label: str = "Mars"
    departure_dv_km_s: float = DEPARTURE_DV_MAGNITUDE_KM_S
    ephemeris: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if not self.ephemeris:
            object.__setattr__(self, "ephemeris", earth_mars_data())
        scale = np.r_[
            np.full(3, self.length_unit * 1000.0), np.full(3, self.velocity_unit * 1000.0)
        ]
        direction = np.asarray(DEPARTURE_DV_DIRECTION) / np.linalg.norm(DEPARTURE_DV_DIRECTION)
        departure_impulse_si = np.r_[np.zeros(3), self.departure_dv_km_s * 1000.0 * direction]
        object.__setattr__(
            self,
            "x0",
            tuple(
                (np.asarray(self.ephemeris["earth_departure_state_si"]) + departure_impulse_si)
                / scale
            ),
        )
        object.__setattr__(
            self, "xf", tuple(np.asarray(self.ephemeris["mars_arrival_state_si"]) / scale)
        )

    @property
    def time_unit(self) -> float:
        return np.sqrt((self.length_unit * 1000.0) ** 3 / self.ephemeris["mu_sun_m3_s2"])


INITIAL_GUESS_SUBSTEPS_PER_INTERVAL = 2

CASE_TYPES = (NrhoL2ToDro, HaloL2ToHaloL1, LyapunovL1ToL2)
CASE_REGISTRY = {kind().test_case_id: kind for kind in CASE_TYPES}
DETERMINISTIC_CASE_REGISTRY = {**CASE_REGISTRY, "earth_mars": EarthMars}


def _compute_root(
    function, lo: float, hi: float, iterations: int = 100, tol: float = 1e-14
) -> float:
    x_symbolic = casadi.MX.sym("x")
    f_symbolic = function(x_symbolic)
    df_symbolic = casadi.jacobian(f_symbolic, x_symbolic)
    f_callable = casadi.Function("root_function", [x_symbolic], [f_symbolic])
    df_function = casadi.Function("root_derivative", [x_symbolic], [df_symbolic])

    f_lo = float(f_callable(lo))
    f_hi = float(f_callable(hi))
    if f_lo * f_hi > 0.0:
        raise ValueError(f"Root is not bracketed on [{lo}, {hi}].")

    x = 0.5 * (lo + hi)
    for _ in range(iterations):
        f_x = float(f_callable(x))
        if abs(f_x) < tol:
            return float(x)

        df_x = float(df_function(x))
        x_new = x - f_x / df_x if abs(df_x) > 1e-15 else 0.5 * (lo + hi)
        if not np.isfinite(x_new) or x_new <= lo or x_new >= hi:
            x_new = 0.5 * (lo + hi)

        f_new = float(f_callable(x_new))
        if f_lo * f_new <= 0.0:
            hi = x_new
            f_hi = f_new
        else:
            lo = x_new
            f_lo = f_new

        if abs(f_new) < tol or abs(hi - lo) < tol:
            return float(x_new)
        x = x_new

    return float(0.5 * (lo + hi))


def get_collinear_lagrange_points(case: CR3BPEarthMoon) -> dict[str, float]:
    def equilibrium_condition(x):
        r1 = casadi.fabs(x + case.mu)
        r2 = casadi.fabs(x - (1.0 - case.mu))
        return x - (1.0 - case.mu) * (x + case.mu) / r1**3 - case.mu * (x - (1.0 - case.mu)) / r2**3

    eps = 1e-9
    return {
        "L1": _compute_root(equilibrium_condition, -case.mu + eps, 1.0 - case.mu - eps),
        "L2": _compute_root(equilibrium_condition, 1.0 - case.mu + eps, 1.5),
        "L3": _compute_root(equilibrium_condition, -1.5, -case.mu - eps),
    }


def eom(case: TestCase, state, control, sigma):
    """Dynamics with explicitly supplied mass-flow thrust magnitude.

    Deterministic optimisation supplies Tmax * throttle; physical diagnostics
    supply the norm of the applied thrust. All inputs use the case scales.
    """

    if any(isinstance(v, (casadi.MX, casadi.SX, casadi.DM)) for v in (state, control, sigma)):
        rx, ry, rz = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        mass = state[6]
        ux, uy, uz = control[0], control[1], control[2]
    else:
        rx, ry, rz, vx, vy, vz, mass = np.asarray(state, dtype=float)
        ux, uy, uz = np.asarray(control, dtype=float)
        sigma = float(sigma)

    if getattr(case, "dynamics_model", "cr3bp") == "heliocentric":
        radius = np.sqrt(rx**2 + ry**2 + rz**2)
        values = (
            vx,
            vy,
            vz,
            -rx / radius**3 + ux / mass,
            -ry / radius**3 + uy / mass,
            -rz / radius**3 + uz / mass,
            -sigma / case.exhaust_velocity_nd,
        )
        if any(isinstance(v, (casadi.MX, casadi.SX, casadi.DM)) for v in (state, control, sigma)):
            return casadi.vertcat(*values)
        return np.asarray(values, dtype=float)

    r1 = np.sqrt((rx + case.mu) ** 2 + ry**2 + rz**2)
    r2 = np.sqrt((rx - (1.0 - case.mu)) ** 2 + ry**2 + rz**2)

    ax = (
        2.0 * vy
        + rx
        - (1.0 - case.mu) * (rx + case.mu) / r1**3
        - case.mu * (rx - (1.0 - case.mu)) / r2**3
        + ux / mass
    )
    ay = -2.0 * vx + ry - (1.0 - case.mu) * ry / r1**3 - case.mu * ry / r2**3 + uy / mass
    az = -(1.0 - case.mu) * rz / r1**3 - case.mu * rz / r2**3 + uz / mass
    # Mass flow through the thrust-magnitude variable rather than ||u||: linear
    # in sigma, hence smooth with zero curvature everywhere. The transcriptions
    # supply Tmax*alpha in the throttle/direction formulation. Legacy callers
    # can still supply an explicit mass-flow magnitude through this shared API.
    mdot = -sigma / case.exhaust_velocity_nd
    if any(isinstance(v, (casadi.MX, casadi.SX, casadi.DM)) for v in (state, control, sigma)):
        return casadi.vertcat(vx, vy, vz, ax, ay, az, mdot)
    return np.array([vx, vy, vz, ax, ay, az, mdot], dtype=float)


def propagate_periodic_orbit(
    case: CR3BPEarthMoon,
    initial_state: np.ndarray,
    period_nd: float,
    steps: int = 1600,
) -> np.ndarray:
    orbit = np.empty((7, steps + 1), dtype=float)
    orbit[:, 0] = initial_state
    h = period_nd / steps
    for k in range(steps):
        orbit[:, k + 1] = integrator.rk7(case, orbit[:, k], np.zeros(3), 0.0, h)
    return orbit


def integrate_with_dense_control(
    case: TestCase,
    initial_state: np.ndarray,
    t_grid_nd: np.ndarray,
    u_values: np.ndarray,
    sigma_values: np.ndarray,
) -> np.ndarray:

    if u_values.shape[1] != t_grid_nd.size or sigma_values.shape[1] != t_grid_nd.size:
        raise ValueError("Dense ZOH control, sigma, and time arrays must have the same length.")

    x_integrated = np.empty((initial_state.size, t_grid_nd.size), dtype=float)
    x_integrated[:, 0] = initial_state

    for k in range(t_grid_nd.size - 1):
        h = float(t_grid_nd[k + 1] - t_grid_nd[k])
        x_integrated[:, k + 1] = integrator.rk7(
            case,
            x_integrated[:, k],
            u_values[:, k],
            float(sigma_values[0, k]),
            h,
        )
    return x_integrated


def rollout_initial_guess(
    case: TestCase,
    mesh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intervals = mesh.size - 1
    u_component_nd = 1e-4 / case.thrust_unit
    u_guess_nd = np.full((3, intervals), u_component_nd, dtype=float)
    sigma_guess_nd = np.full((1, intervals), np.linalg.norm(u_guess_nd[:, 0]), dtype=float)

    x_guess = np.empty((7, intervals + 1), dtype=float)
    x_guess[:, 0] = case.x0_augmented_state
    for k in range(intervals):
        h = case.tof_nd * float(mesh[k + 1] - mesh[k])
        substep = h / INITIAL_GUESS_SUBSTEPS_PER_INTERVAL
        state = x_guess[:, k]
        for _ in range(INITIAL_GUESS_SUBSTEPS_PER_INTERVAL):
            state = integrator.rk7(case, state, u_guess_nd[:, k], sigma_guess_nd[0, k], substep)
        x_guess[:, k + 1] = state

    return x_guess, u_guess_nd, sigma_guess_nd
