"""
Compute Earth-Moon CR3BP low-thrust transfers with hp-adaptive Radau collocation.

The continuous-time optimal control problem is discretized with a variable-size
Radau collocation mesh, formulated with CasADi, and solved with IPOPT.
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import multiprocessing as mp
import casadi
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import integrator

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.usetex": True,
    }
)

@dataclass(frozen=True)
class CR3BPEarthMoon:
    gm1: float = 398600.0          # [km^3/s^2]
    gm2: float = 4902.80           # [km^3/s^2]
    length_unit: float = 384399.0  # [km]
    moon_radius: float = 1737.5    # [km]

    m0_wet: float = 1000.0         # [kg]
    m0_dry: float = 500.0          # [kg]
    thrust_max: float = 0.5        # [N]
    isp: float = 2000.0            # [s]

    @property
    def mu(self) -> float:
        return self.gm2 / (self.gm1 + self.gm2)  # [-]

    @property
    def time_unit(self) -> float:
        return np.sqrt(self.length_unit**3 / (self.gm1 + self.gm2))  # [s]

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

    @property
    def moon_radius_lu(self) -> float:
        return self.moon_radius / self.length_unit  # [-]


@dataclass(frozen=True)
class TestCase(CR3BPEarthMoon):
    test_case_id: str = ""       
    display_name: str = ""       
    tof_days: float = 0.0        # [days]
    nodes: int = 0               

    x0: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # [-]
    xf: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # [-]

    departure_label: str = "departure orbit"  
    target_label: str = "target orbit"        
    departure_period_nd: float | None = None    # [-]
    target_period_nd: float | None = None       # [-]
    departure_period_days: float | None = None  # [days]
    target_period_days: float | None = None     # [days]

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
        if self.departure_period_nd is not None:
            return self.departure_period_nd  
        if self.departure_period_days is not None:
            return self.departure_period_days * 86400.0 / self.time_unit  
        return None

    @property
    def target_period_nd(self) -> float | None:
        if self.target_period_nd is not None:
            return self.target_period_nd  # [-]
        if self.target_period_days is not None:
            return self.target_period_days * 86400.0 / self.time_unit  
        return None


@dataclass(frozen=True)
class LyapunovL1ToL2(TestCase):
    test_case_id: str = "lyapunov_l1_to_l2"  
    display_name: str = "Lyapunov L1 to Lyapunov L2"  
    tof_days: float = 12.0  # [days]
    nodes: int = 300        # [-]
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
    departure_period_nd: float = 2.9750964922007723  # [-]
    target_period_nd: float = 3.49306635929003       # [-]


@dataclass(frozen=True)
class HaloL2ToHaloL1(TestCase):
    test_case_id: str = "halo_l2_to_halo_l1"  
    display_name: str = "Halo L2 to Halo L1"  # [-]
    tof_days: float = 20.0  # [days]
    nodes: int = 300        # [-]
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
    departure_period_nd: float = 3.2746644337639852  # [-]
    target_period_nd: float = 2.5748200748171399     # [-]


@dataclass
class OCPSolution:
    mesh: np.ndarray
    degrees: np.ndarray
    x: np.ndarray
    u: np.ndarray
    sigma: np.ndarray
    stages: dict[tuple[int, int], np.ndarray]
    diagnostics: dict[str, float]



CASE_TYPES: tuple[type[TestCase], ...] = (LyapunovL1ToL2, HaloL2ToHaloL1)
CASE_REGISTRY: dict[str, type[TestCase]] = {
    case_type().test_case_id: case_type for case_type in CASE_TYPES
}
MAX_ITER = 10000
PRINT_LEVEL = 0
TOL = 1e-8
DEFAULT_OUTPUT_DIR = Path("output/cr3bp")
DEFAULT_OUTPUT_PREFIX = ""
INITIAL_GUESS = OCPSolution | tuple[np.ndarray, np.ndarray, np.ndarray] | None
COLLOCATION_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _compute_root(function, lo: float, hi: float, iterations: int = 100, tol: float = 1e-14) -> float:
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

def collinear_lagrange_points(case: CR3BPEarthMoon) -> dict[str, float]:
    def equilibrium_condition(x):
        r1 = casadi.fabs(x + case.mu)
        r2 = casadi.fabs(x - (1.0 - case.mu))
        return (
            x
            - (1.0 - case.mu) * (x + case.mu) / r1**3
            - case.mu * (x - (1.0 - case.mu)) / r2**3
        )

    eps = 1e-9
    return {
        "L1": _compute_root(equilibrium_condition, -case.mu + eps, 1.0 - case.mu - eps),
        "L2": _compute_root(equilibrium_condition, 1.0 - case.mu + eps, 1.5),
        "L3": _compute_root(equilibrium_condition, -1.5, -case.mu - eps),
    }

def eom(case: CR3BPEarthMoon, state, control, sigma):
    if any(isinstance(v, (casadi.MX, casadi.SX, casadi.DM)) for v in (state, control, sigma)):        
        rx, ry, rz = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        mass = state[6]
        ux, uy, uz = control[0], control[1], control[2]
        sqrt = casadi.sqrt
    else:
        rx, ry, rz, vx, vy, vz, mass = np.asarray(state, dtype=float)
        ux, uy, uz = np.asarray(control, dtype=float)
        sigma = float(sigma)
        sqrt = np.sqrt

    r1 = sqrt((rx + case.mu) ** 2 + ry**2 + rz**2)
    r2 = sqrt((rx - (1.0 - case.mu)) ** 2 + ry**2 + rz**2)

    ax = (
        2.0 * vy
        + rx
        - (1.0 - case.mu) * (rx + case.mu) / r1**3
        - case.mu * (rx - (1.0 - case.mu)) / r2**3
        + ux / mass
    )
    ay = (
        -2.0 * vx
        + ry
        - (1.0 - case.mu) * ry / r1**3
        - case.mu * ry / r2**3
        + uy / mass
    )
    az = -(1.0 - case.mu) * rz / r1**3 - case.mu * rz / r2**3 + uz / mass
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
        orbit[:, k + 1] = integrator.rk4(case, orbit[:, k], np.zeros(3), 0.0, h)
    return orbit

def _rollout(
    case: TestCase,
    nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_component_nd = 1e-6 / case.thrust_unit
    u_guess = np.full((3, nodes + 1), u_component_nd, dtype=float)
    sigma_guess = np.full((1, nodes + 1), np.linalg.norm(u_guess[:, 0]), dtype=float)
    x_guess = np.empty((7, nodes + 1), dtype=float)
    x_guess[:, 0] = case.x0_augmented_state
    h = case.tof_nd / nodes
    for k in range(nodes):
        x_guess[:, k + 1] = integrator.rk4(case, x_guess[:, k], u_guess[:, k], sigma_guess[0, k], h)
    return x_guess, u_guess, sigma_guess

def _resample_solution(
    sol: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    nodes: int,
    case: TestCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sol is None:
        return _rollout(case, nodes)
    old_x, old_u, old_sigma = sol
    old_grid = np.linspace(0.0, 1.0, old_u.shape[1])
    new_grid = np.linspace(0.0, 1.0, nodes + 1)
    x_guess = np.vstack([np.interp(new_grid, old_grid, old_x[i]) for i in range(old_x.shape[0])])
    u_guess = np.vstack([np.interp(new_grid, old_grid, old_u[i]) for i in range(old_u.shape[0])])
    sigma_guess = np.vstack([np.interp(new_grid, old_grid, old_sigma[0])])
    return x_guess, u_guess, sigma_guess

def _warm_start(
    case: TestCase,
    nodes: int,
    sol: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    max_iter: int,
    print_level: int,
    extra_options: dict[str, float | int | str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    "Solve the optimal control problem with a Hermite-Simpson collocation method "
    "to warm-start the hp-adaptive Radau refinement."
    opti = casadi.Opti()
    h = case.tof_nd / nodes
    x_var = opti.variable(7, nodes + 1)
    u_var = opti.variable(3, nodes + 1)
    sigma_var = opti.variable(1, nodes + 1)
    fuel_consumed_nd = 0.0
    max_thrust_nd = case.max_thrust_nd

    for k in range(nodes + 1):
        opti.subject_to(casadi.dot(u_var[:, k], u_var[:, k]) <= sigma_var[0, k] ** 2)
        opti.subject_to(opti.bounded(0.0, sigma_var[0, k], max_thrust_nd))
        opti.subject_to(x_var[6, k] >= case.m0_dry_nd)

    for k in range(nodes):
        f_k = eom(case, x_var[:, k], u_var[:, k], sigma_var[0, k])
        f_kp1 = eom(case, x_var[:, k + 1], u_var[:, k + 1], sigma_var[0, k + 1])
        x_mid = 0.5 * (x_var[:, k] + x_var[:, k + 1]) + h / 8.0 * (f_k - f_kp1)
        u_mid = 0.5 * (u_var[:, k] + u_var[:, k + 1])
        sigma_mid = 0.5 * (sigma_var[0, k] + sigma_var[0, k + 1])
        f_mid = eom(case, x_mid, u_mid, sigma_mid)
        opti.subject_to(x_var[:, k + 1] - x_var[:, k] == h / 6.0 * (f_k + 4.0 * f_mid + f_kp1))
        fuel_consumed_nd += (
            h
            / 6.0
            * (sigma_var[0, k] + 4.0 * sigma_mid + sigma_var[0, k + 1])
            / case.exhaust_velocity_nd
        )

    opti.subject_to(x_var[:, 0] == case.x0_augmented_state)
    opti.subject_to(x_var[0:6, nodes] == case.xf_state)
    fuel_consumed = fuel_consumed_nd * case.m0_wet
    opti.minimize(fuel_consumed)

    x_guess, u_guess, sigma_guess = _resample_solution(sol, nodes, case)
    sigma_guess = np.minimum(np.maximum(sigma_guess, 1e-7), max_thrust_nd)

    opti.set_initial(x_var, x_guess)
    opti.set_initial(u_var, u_guess)
    opti.set_initial(sigma_var, sigma_guess)

    ipopt_settings = {
        "max_iter": max_iter,
        # Desired convergence tolerance (relative)
        "tol": TOL,                     
        # "Acceptable" convergence tolerance (relative)
        "acceptable_tol": TOL,
        # Desired threshold for the constraint and variable bound violation
        "constr_viol_tol": TOL,
        # Desired threshold for the dual infeasibility
        "dual_inf_tol": TOL,
        # Desired threshold for the complementarity conditions
        "compl_inf_tol": TOL,
        # Update strategy for barrier parameter.
        "mu_strategy": "adaptive",
        "print_level": print_level,
        # Suppresses IPOPT banner
        "sb": "yes",
    }
    if extra_options:
        ipopt_settings.update(extra_options)
    # NOTE: Expand the MX graph into scalar SX operations before solving, which can speed up
    # numerical evaluations significantly but may increase memory usage.
    opti.solver("ipopt", {"expand": True}, ipopt_settings)

    sol = opti.solve()
    x_sol = np.asarray(sol.value(x_var), dtype=float)
    u_sol = np.asarray(sol.value(u_var), dtype=float)
    sigma_sol = np.asarray(sol.value(sigma_var), dtype=float).reshape(1, -1)
    diagnostics = get_diagnostics(case, x_sol, u_sol, sigma_sol)
    diagnostics["fuel_consumed_kg"] = float(sol.value(fuel_consumed))
    diagnostics["nodes"] = float(nodes)
    return x_sol, u_sol, sigma_sol, diagnostics

def integrate_with_dense_control(
    case: TestCase,
    initial_state: np.ndarray,
    t_grid_nd: np.ndarray,
    u_values: np.ndarray,
    sigma_values: np.ndarray,
) -> np.ndarray:
    x_integrated = np.empty((initial_state.size, t_grid_nd.size), dtype=float)
    x_integrated[:, 0] = initial_state

    def rhs(t_nd: float, state: np.ndarray) -> np.ndarray:
        control = np.array([np.interp(t_nd, t_grid_nd, u_values[i]) for i in range(u_values.shape[0])])
        sigma = float(np.interp(t_nd, t_grid_nd, sigma_values[0]))
        control_norm = np.linalg.norm(control)
        return eom(case, state, control, control_norm)

    for k in range(t_grid_nd.size - 1):
        t_k = float(t_grid_nd[k])
        h = float(t_grid_nd[k + 1] - t_grid_nd[k])
        state = x_integrated[:, k]
        k1 = rhs(t_k, state)
        k2 = rhs(t_k + 0.5 * h, state + 0.5 * h * k1)
        k3 = rhs(t_k + 0.5 * h, state + 0.5 * h * k2)
        k4 = rhs(t_k + h, state + h * k3)
        x_integrated[:, k + 1] = state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return x_integrated

def compute_augm_state_error(
    case: TestCase,
    t_dense_days: np.ndarray,
    x_dense: np.ndarray,
    u_dense: np.ndarray,
    sigma_dense: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_dense_nd = t_dense_days * 86400.0 / case.time_unit
    x_integrated = integrate_with_dense_control(case, x_dense[:, 0], t_dense_nd, u_dense, sigma_dense)
    position_error_lu = np.linalg.norm(x_integrated[0:3] - x_dense[0:3], axis=0)
    position_error_m = position_error_lu * case.length_unit * 1000.0
    velocity_error_nd = np.linalg.norm(x_integrated[3:6] - x_dense[3:6], axis=0)
    velocity_error_m_per_s = velocity_error_nd * case.velocity_unit * 1000.0
    optimized_mass_consumed_kg = (x_dense[6, 0] - x_dense[6]) * case.m0_wet
    integrated_mass_consumed_kg = (x_integrated[6, 0] - x_integrated[6]) * case.m0_wet
    mass_consumption_difference_kg = integrated_mass_consumed_kg - optimized_mass_consumed_kg
    mass_consumption_abs_difference_kg = np.abs(mass_consumption_difference_kg)
    return (
        x_integrated,
        position_error_lu,
        position_error_m,
        velocity_error_nd,
        velocity_error_m_per_s,
        mass_consumption_difference_kg,
        mass_consumption_abs_difference_kg,
    )

def get_diagnostics(
    case: TestCase,
    x_sol: np.ndarray,
    u_sol: np.ndarray,
    sigma_sol: np.ndarray,
) -> dict[str, float]:
    u_norm = np.linalg.norm(u_sol, axis=0)
    final_error = np.max(np.abs(x_sol[:6, -1] - case.xf_state))
    return {
        "final_mass_kg": x_sol[6, -1] * case.m0_wet,
        "max_thrust": float(np.max(u_norm) * case.thrust_unit),
        "max_sigma": float(np.max(sigma_sol) * case.thrust_unit),
        "max_sigma_minus_norm_n": float(np.max(sigma_sol[0] - u_norm) * case.thrust_unit),
        "terminal_error_nd": float(final_error),
    }

@dataclass(frozen=True)
class HPAdaptiveOptions:
    initial_intervals: int = 150
    min_degree: int = 3
    max_degree: int = 10
    max_intervals: int = 1000
    max_adapt_iterations: int = 10
    defect_tolerance: float = 1e-10
    split_error_factor: float = 20.0
    control_rel_change_treshold: float = 0.25
    p_coarsen_factor: float = 0.02
    rk4_substeps_per_interval: int = 8


def collocation_coefficients(degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if degree in COLLOCATION_CACHE:
        return COLLOCATION_CACHE[degree]

    tau_root = np.array([0.0] + casadi.collocation_points(degree, "radau"), dtype=float)
    c_matrix = np.zeros((degree + 1, degree + 1), dtype=float)
    d_vector = np.zeros(degree + 1, dtype=float)
    b_vector = np.zeros(degree + 1, dtype=float)

    for r in range(degree + 1):
        basis = np.poly1d([1.0])
        for s in range(degree + 1):
            if s != r:
                basis *= np.poly1d([1.0, -tau_root[s]]) / (tau_root[r] - tau_root[s])

        basis_derivative = np.polyder(basis)
        for j in range(degree + 1):
            c_matrix[r, j] = basis_derivative(tau_root[j])

        basis_integral = np.polyint(basis)
        b_vector[r] = basis_integral(1.0) - basis_integral(0.0)
        d_vector[r] = basis(1.0)

    COLLOCATION_CACHE[degree] = (tau_root, c_matrix, d_vector, b_vector)
    return COLLOCATION_CACHE[degree]


def lagrange_basis_values(theta: float, tau_root: np.ndarray) -> np.ndarray:
    values = np.ones(tau_root.size, dtype=float)
    for r in range(tau_root.size):
        for s in range(tau_root.size):
            if s != r:
                values[r] *= (theta - tau_root[s]) / (tau_root[r] - tau_root[s])
    return values


def variable_rollout_guess(
    case: TestCase,
    mesh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intervals = mesh.size - 1
    u_component_nd = 1e-6 / case.thrust_unit
    u_guess = np.full((3, intervals + 1), u_component_nd, dtype=float)
    sigma_guess = np.full((1, intervals + 1), np.linalg.norm(u_guess[:, 0]), dtype=float)

    x_guess = np.empty((7, intervals + 1), dtype=float)
    x_guess[:, 0] = case.x0_augmented_state
    for k in range(intervals):
        h = case.tof_nd * float(mesh[k + 1] - mesh[k])
        x_guess[:, k + 1] = integrator.rk4(
            case,
            x_guess[:, k],
            u_guess[:, k],
            sigma_guess[0, k],
            h,
        )

    return x_guess, u_guess, sigma_guess


def evaluate_hp_state(OCPSolution: OCPSolution, fraction: float) -> np.ndarray:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction <= OCPSolution.mesh[0]:
        return OCPSolution.x[:, 0].copy()
    if fraction >= OCPSolution.mesh[-1]:
        return OCPSolution.x[:, -1].copy()

    interval = int(np.searchsorted(OCPSolution.mesh, fraction, side="right") - 1)
    interval = min(max(interval, 0), OCPSolution.degrees.size - 1)
    left = float(OCPSolution.mesh[interval])
    right = float(OCPSolution.mesh[interval + 1])
    theta = (fraction - left) / (right - left)
    degree = int(OCPSolution.degrees[interval])
    tau_root, _, _, _ = collocation_coefficients(degree)
    interval_states = [OCPSolution.x[:, interval]]
    for j in range(1, degree):
        interval_states.append(OCPSolution.stages[(interval, j)])
    interval_states.append(OCPSolution.x[:, interval + 1])

    basis = lagrange_basis_values(theta, tau_root)
    state = np.zeros(7, dtype=float)
    for coefficient, interval_state in zip(basis, interval_states):
        state += coefficient * interval_state
    return state


def evaluate_guess_state(case: TestCase, guess: INITIAL_GUESS, fraction: float) -> np.ndarray:
    if guess is None:
        return (1.0 - fraction) * case.x0_augmented_state + fraction * case.xf_augmented_state

    if isinstance(guess, OCPSolution):
        return evaluate_hp_state(guess, fraction)

    old_x, _, _ = guess
    old_grid = np.linspace(0.0, 1.0, old_x.shape[1])
    return np.array([np.interp(fraction, old_grid, old_x[i]) for i in range(old_x.shape[0])])


def sample_endpoint_guess(
    case: TestCase,
    guess: INITIAL_GUESS,
    mesh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if guess is None:
        return variable_rollout_guess(case, mesh)

    if isinstance(guess, OCPSolution):
        x_guess = np.column_stack([evaluate_hp_state(guess, fraction) for fraction in mesh])
        u_guess = np.vstack([np.interp(mesh, guess.mesh, guess.u[i]) for i in range(guess.u.shape[0])])
        sigma_guess = np.vstack([np.interp(mesh, guess.mesh, guess.sigma[0])])
        return x_guess, u_guess, sigma_guess

    old_x, old_u, old_sigma = guess
    old_grid = np.linspace(0.0, 1.0, old_u.shape[1])
    x_guess = np.vstack([np.interp(mesh, old_grid, old_x[i]) for i in range(old_x.shape[0])])
    u_guess = np.vstack([np.interp(mesh, old_grid, old_u[i]) for i in range(old_u.shape[0])])
    sigma_guess = np.vstack([np.interp(mesh, old_grid, old_sigma[0])])
    return x_guess, u_guess, sigma_guess


def uniform_tuple_guess(
    case: TestCase,
    guess: INITIAL_GUESS,
    nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if guess is None:
        return None
    mesh = np.linspace(0.0, 1.0, nodes + 1)
    return sample_endpoint_guess(case, guess, mesh)


def solve_hp(
    case: TestCase,
    mesh: np.ndarray,
    degrees: np.ndarray,
    guess: INITIAL_GUESS,
    max_iter: int,
    print_level: int,
    clip_control_guess: bool,
    clip_fraction: float = 0.95,
) -> OCPSolution:
    opti = casadi.Opti()
    intervals = degrees.size
    thrust_cap = case.max_thrust_nd
    x_var = opti.variable(7, intervals + 1)
    u_var = opti.variable(3, intervals + 1)
    sigma_var = opti.variable(1, intervals + 1)
    stage_vars: list[tuple[int, int, casadi.MX]] = []
    fuel_consumed_nd = 0.0

    for k in range(intervals + 1):
        opti.subject_to(casadi.dot(u_var[:, k], u_var[:, k]) <= sigma_var[0, k] ** 2)
        opti.subject_to(opti.bounded(0.0, sigma_var[0, k], thrust_cap))
        opti.subject_to(x_var[6, k] >= case.m0_dry_nd)

    for k in range(intervals):
        degree = int(degrees[k])
        tau_root, c_matrix, _, b_vector = collocation_coefficients(degree)
        h = case.tof_nd * float(mesh[k + 1] - mesh[k])
        interval_states = [x_var[:, k]]

        for j in range(1, degree):
            x_stage = opti.variable(7)
            opti.subject_to(x_stage[6] >= case.m0_dry_nd)
            stage_vars.append((k, j, x_stage))
            interval_states.append(x_stage)
        interval_states.append(x_var[:, k + 1])

        for j in range(1, degree + 1):
            tau_j = float(tau_root[j])
            x_j = interval_states[j]
            u_j = (1.0 - tau_j) * u_var[:, k] + tau_j * u_var[:, k + 1]
            sigma_j = (1.0 - tau_j) * sigma_var[0, k] + tau_j * sigma_var[0, k + 1]

            polynomial_derivative = c_matrix[0, j] * interval_states[0]
            for r in range(1, degree + 1):
                polynomial_derivative += c_matrix[r, j] * interval_states[r]

            f_j = eom(case, x_j, u_j, sigma_j)
            opti.subject_to(polynomial_derivative == h * f_j)
            fuel_consumed_nd += h * b_vector[j] * sigma_j / case.exhaust_velocity_nd

    opti.subject_to(x_var[:, 0] == case.x0_augmented_state)
    opti.subject_to(x_var[0:6, intervals] == case.xf_state)

    fuel_consumed = fuel_consumed_nd * case.m0_wet
    opti.minimize(fuel_consumed)

    x_guess, u_guess, sigma_guess = sample_endpoint_guess(case, guess, mesh)
    sigma_guess = np.minimum(np.maximum(sigma_guess, 1e-7), clip_fraction * thrust_cap)
    if clip_control_guess:
        for k in range(intervals + 1):
            norm_u = np.linalg.norm(u_guess[:, k])
            if norm_u > 0.98 * sigma_guess[0, k]:
                u_guess[:, k] *= 0.98 * sigma_guess[0, k] / norm_u

    opti.set_initial(x_var, x_guess)
    opti.set_initial(u_var, u_guess)
    opti.set_initial(sigma_var, sigma_guess)
    for k, j, x_stage in stage_vars:
        degree = int(degrees[k])
        tau_root, _, _, _ = collocation_coefficients(degree)
        fraction = float(mesh[k] + tau_root[j] * (mesh[k + 1] - mesh[k]))
        if guess is None:
            stage_guess = (1.0 - tau_root[j]) * x_guess[:, k] + tau_root[j] * x_guess[:, k + 1]
        else:
            stage_guess = evaluate_guess_state(case, guess, fraction)
        opti.set_initial(x_stage, stage_guess)

    ipopt_options: dict[str, float | int | str] = {
        "max_iter": max_iter,
        "tol": TOL,
        "acceptable_tol": TOL,
        "constr_viol_tol": TOL,
        "dual_inf_tol": TOL,
        "compl_inf_tol": TOL,
        "mu_strategy": "adaptive",
        "print_level": print_level,
        "sb": "yes",
    }

    opti.solver("ipopt", {"expand": True}, ipopt_options)

    sol = opti.solve()
    x_sol = np.asarray(sol.value(x_var), dtype=float)
    u_sol = np.asarray(sol.value(u_var), dtype=float)
    sigma_sol = np.asarray(sol.value(sigma_var), dtype=float).reshape(1, -1)
    stages = {
        (k, j): np.asarray(sol.value(x_stage), dtype=float).reshape(7)
        for k, j, x_stage in stage_vars
    }

    diagnostics = get_diagnostics(case, x_sol, u_sol, sigma_sol)
    diagnostics["fuel_consumed_kg"] = float(sol.value(fuel_consumed))
    diagnostics["intervals"] = float(intervals)
    diagnostics["total_collocation_points"] = float(np.sum(degrees))
    diagnostics["min_degree"] = float(np.min(degrees))
    diagnostics["max_degree"] = float(np.max(degrees))
    diagnostics["mean_degree"] = float(np.mean(degrees))
    diagnostics["min_step_days"] = float(np.min(np.diff(mesh)) * case.tof_days)
    diagnostics["max_step_days"] = float(np.max(np.diff(mesh)) * case.tof_days)

    return OCPSolution(
        mesh=np.asarray(mesh, dtype=float).copy(),
        degrees=np.asarray(degrees, dtype=int).copy(),
        x=x_sol,
        u=u_sol,
        sigma=sigma_sol,
        stages=stages,
        diagnostics=diagnostics,
    )


def solve_hp_with_retries(
    case: TestCase,
    mesh: np.ndarray,
    degrees: np.ndarray,
    guess: INITIAL_GUESS,
    options: HPAdaptiveOptions,
    max_iter: int,
    print_level: int,
    log_prefix: str,
    prefer_hermite_simpson_warm_start: bool = False,
) -> tuple[OCPSolution, bool]:
    def hermite_simpson_warm_start() -> OCPSolution:
        print(f"{log_prefix}  retry: hermite-simpson warm start", flush=True)
        nodes = mesh.size - 1
        hs_guess = uniform_tuple_guess(case, guess, nodes)
        hs_x, hs_u, hs_sigma, _ = _warm_start(
            case=case,
            nodes=nodes,
            sol=hs_guess,
            max_iter=max_iter,
            print_level=print_level,
            clip_control_guess=guess is None,
            clip_fraction=0.99,
        )
        return solve_hp(
            case=case,
            mesh=mesh,
            degrees=degrees,
            guess=(hs_x, hs_u, hs_sigma),
            max_iter=max_iter,
            print_level=print_level,
            clip_control_guess=False,
            clip_fraction=0.99,
        )

    if prefer_hermite_simpson_warm_start:
        try:
            return hermite_simpson_warm_start(), True
        except RuntimeError:
            print(f"{log_prefix}  warm start failed; falling back to direct attempts", flush=True)

    attempts = [
        ("default", guess is None, 0.95, {}),
        ("preserve-control", False, 0.99, {}),
        ("lower-sigma", True, 0.60, {}),
        ("fixed-mu", guess is None, 0.95, {"mu_strategy": "monotone"}),
    ]
    last_error: Exception | None = None
    for label, attempt_clip, sigma_fraction, extra_options in attempts:
        try:
            if label != "default":
                print(f"{log_prefix}  retry: {label}", flush=True)
            return (
                solve_hp(
                    case=case,
                    mesh=mesh,
                    degrees=degrees,
                    guess=guess,
                    max_iter=max_iter,
                    print_level=print_level,
                    clip_control_guess=attempt_clip,
                    clip_fraction=sigma_fraction,
                ),
                False,
            )
        except RuntimeError as exc:
            last_error = exc

    try:
        return hermite_simpson_warm_start(), True
    except RuntimeError as exc:
        if last_error is not None:
            raise last_error from exc
        raise


def integrate_interval_rk4(
    case: TestCase,
    x0: np.ndarray,
    u0: np.ndarray,
    u1: np.ndarray,
    sigma0: float,
    sigma1: float,
    h: float,
    substeps: int,
) -> np.ndarray:
    state = np.asarray(x0, dtype=float).copy()
    dt = h / substeps

    def rhs(theta: float, local_state: np.ndarray) -> np.ndarray:
        control = (1.0 - theta) * u0 + theta * u1
        sigma = (1.0 - theta) * sigma0 + theta * sigma1
        return eom(case, local_state, control, sigma)

    for s in range(substeps):
        theta = s / substeps
        half_step = 0.5 / substeps
        full_step = 1.0 / substeps
        k1 = rhs(theta, state)
        k2 = rhs(theta + half_step, state + 0.5 * dt * k1)
        k3 = rhs(theta + half_step, state + 0.5 * dt * k2)
        k4 = rhs(theta + full_step, state + dt * k3)
        state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return state


def estimate_interval_defects(
    case: TestCase,
    OCPSolution: OCPSolution,
    substeps: int,
) -> dict[str, np.ndarray]:
    intervals = OCPSolution.degrees.size
    scaled_error = np.empty(intervals, dtype=float)
    position_error_m = np.empty(intervals, dtype=float)
    velocity_error_m_per_s = np.empty(intervals, dtype=float)
    mass_error_kg = np.empty(intervals, dtype=float)

    for k in range(intervals):
        h = case.tof_nd * float(OCPSolution.mesh[k + 1] - OCPSolution.mesh[k])
        x_integrated = integrate_interval_rk4(
            case,
            OCPSolution.x[:, k],
            OCPSolution.u[:, k],
            OCPSolution.u[:, k + 1],
            float(OCPSolution.sigma[0, k]),
            float(OCPSolution.sigma[0, k + 1]),
            h,
            substeps,
        )
        error = x_integrated - OCPSolution.x[:, k + 1]
        scale = np.maximum(1.0, np.maximum(np.abs(OCPSolution.x[:, k]), np.abs(OCPSolution.x[:, k + 1])))
        scaled_error[k] = float(np.max(np.abs(error) / scale))
        position_error_m[k] = float(np.linalg.norm(error[0:3]) * case.length_unit * 1000.0)
        velocity_error_m_per_s[k] = float(np.linalg.norm(error[3:6]) * case.velocity_unit * 1000.0)
        mass_error_kg[k] = float(abs(error[6]) * case.m0_wet)

    return {
        "scaled": scaled_error,
        "position_m": position_error_m,
        "velocity_m_per_s": velocity_error_m_per_s,
        "mass_kg": mass_error_kg,
    }


def refine_hp_mesh(
    OCPSolution: OCPSolution,
    defects: np.ndarray,
    options: HPAdaptiveOptions,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mesh = OCPSolution.mesh
    degrees = OCPSolution.degrees
    intervals = degrees.size
    over_tol = defects > options.defect_tolerance

    split_candidates: list[int] = []
    for k in np.flatnonzero(over_tol):
        degree = int(degrees[k])
        u0 = OCPSolution.u[:, k]
        u1 = OCPSolution.u[:, k + 1]
        jump = np.linalg.norm(u1 - u0) / max(np.linalg.norm(u0), np.linalg.norm(u1), 1e-14)
        should_split = (
            degree >= options.max_degree
            or defects[k] > options.split_error_factor * options.defect_tolerance
            or jump > options.control_rel_change_treshold

        )
        if should_split:
            split_candidates.append(int(k))

    remaining_capacity = max(options.max_intervals - intervals, 0)
    split_selected = set(
        sorted(split_candidates, key=lambda idx: defects[idx], reverse=True)[:remaining_capacity]
    )

    new_mesh = [float(mesh[0])]
    new_degrees: list[int] = []
    p_increases = 0
    p_decreases = 0

    for k in range(intervals):
        degree = int(degrees[k])
        if k in split_selected:
            midpoint = 0.5 * float(mesh[k] + mesh[k + 1])
            new_degrees.append(degree)
            new_mesh.append(midpoint)
            new_degrees.append(degree)
            new_mesh.append(float(mesh[k + 1]))
            continue

        new_degree = degree
        if over_tol[k] and new_degree < options.max_degree:
            new_degree += 1
            p_increases += 1
        elif (
            defects[k] < options.p_coarsen_factor * options.defect_tolerance
            and new_degree > options.min_degree
        ):
            new_degree -= 1
            p_decreases += 1

        new_degrees.append(new_degree)
        new_mesh.append(float(mesh[k + 1]))

    new_mesh_array = np.asarray(new_mesh, dtype=float)
    new_degrees_array = np.asarray(new_degrees, dtype=int)
    changed = len(split_selected) + p_increases + p_decreases
    summary = {
        "split_count": float(len(split_selected)),
        "p_increase_count": float(p_increases),
        "p_decrease_count": float(p_decreases),
        "changed_count": float(changed),
    }
    return new_mesh_array, new_degrees_array, summary


def interpolate_hp_OCPSolution(
    case: TestCase,
    OCPSolution: OCPSolution,
    samples_per_interval: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_days_values: list[float] = []
    x_values: list[np.ndarray] = []
    u_values: list[np.ndarray] = []
    sigma_values: list[float] = []

    for k, degree in enumerate(OCPSolution.degrees):
        degree = int(degree)
        tau_root, _, _, _ = collocation_coefficients(degree)
        interval_states = [OCPSolution.x[:, k]]
        for j in range(1, degree):
            interval_states.append(OCPSolution.stages[(k, j)])
        interval_states.append(OCPSolution.x[:, k + 1])

        theta_grid = np.linspace(0.0, 1.0, samples_per_interval + 1)
        thetas = theta_grid if k == 0 else theta_grid[1:]
        for theta in thetas:
            theta = float(theta)
            basis = lagrange_basis_values(theta, tau_root)
            x_theta = np.zeros(7, dtype=float)
            for coefficient, interval_state in zip(basis, interval_states):
                x_theta += coefficient * interval_state

            u_theta = (1.0 - theta) * OCPSolution.u[:, k] + theta * OCPSolution.u[:, k + 1]
            sigma_theta = (1.0 - theta) * OCPSolution.sigma[0, k] + theta * OCPSolution.sigma[0, k + 1]
            fraction = float(OCPSolution.mesh[k] + theta * (OCPSolution.mesh[k + 1] - OCPSolution.mesh[k]))

            t_days_values.append(fraction * case.tof_days)
            x_values.append(x_theta)
            u_values.append(u_theta)
            sigma_values.append(float(sigma_theta))

    return (
        np.asarray(t_days_values, dtype=float),
        np.column_stack(x_values),
        np.column_stack(u_values),
        np.asarray(sigma_values, dtype=float).reshape(1, -1),
    )


AXIS_LABELS = ("x [LU]", "y [LU]", "z [LU]")
BLACK_COLOR = "#000000"
TURQUOISE_COLOR = "#00A6A6"
DARK_RED_COLOR = "#8B0000"
DARK_RED_VARIANT_COLOR = "#6F0000"
GREY_COLOR = "#8A8A8A"
DARK_GREY_COLOR = "#2F2F2F"
LIGHT_GREY_COLOR = "#D8D8D8"
TRAJECTORY_COLOR = BLACK_COLOR
THRUST_COLOR = DARK_RED_COLOR
SECONDARY_DATA_COLOR = TURQUOISE_COLOR
REFERENCE_COLOR = GREY_COLOR
POINT_MARKER_COLOR = TURQUOISE_COLOR
MOON_COLOR = GREY_COLOR
LIMIT_COLOR = DARK_RED_VARIANT_COLOR
FIGURE_DPI = 220
SQUARE_FIGSIZE = (5.6, 5.6)
WIDE_FIGSIZE = SQUARE_FIGSIZE
THREE_D_FIGSIZE = (5.8, 5.8)
AXIS_LABEL_FONT_SIZE = 12
TICK_LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10
ANNOTATION_FONT_SIZE = 9
AXIS_SPINE_WIDTH = 1.25
MAJOR_TICK_WIDTH = 1.15
MINOR_TICK_WIDTH = 0.9
TRAJECTORY_LINE_WIDTH = 1.8
REFERENCE_LINE_WIDTH = 1.2
GUIDE_LINE_WIDTH = 1.2
PROJECTION_COLORS = {
    (0, 1): TRAJECTORY_COLOR,
    (0, 2): TRAJECTORY_COLOR,
    (1, 2): TRAJECTORY_COLOR,
}
PROJECTION_AXES = ((0, 1), (0, 2), (1, 2))
LYAPUNOV_L1_TO_L2_PROJECTION_AXES = ((0, 1),)
PROJECTION_SUFFIXES = {
    (0, 1): "xy",
    (0, 2): "xz",
    (1, 2): "yz",
}
CALEB_TRAJECTORY_COLOR = TRAJECTORY_COLOR
CALEB_REFERENCE_COLOR = REFERENCE_COLOR
CALEB_THRUST_COLOR = THRUST_COLOR
CALEB_POINT_COLOR = POINT_MARKER_COLOR
CALEB_MOON_COLOR = MOON_COLOR


def style_2d_axis(ax, *, equal_axis: bool = False, grid_which: str = "major") -> None:
    ax.minorticks_on()
    ax.grid(True, which=grid_which, color=LIGHT_GREY_COLOR, alpha=0.75, linewidth=0.7)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        top=False,
        right=False,
        colors=DARK_GREY_COLOR,
        labelsize=TICK_LABEL_FONT_SIZE,
        width=MAJOR_TICK_WIDTH,
        length=5.5,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        top=False,
        right=False,
        colors=DARK_GREY_COLOR,
        width=MINOR_TICK_WIDTH,
        length=3.0,
    )
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.title.set_size(AXIS_LABEL_FONT_SIZE)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(GREY_COLOR)
        spine.set_linewidth(AXIS_SPINE_WIDTH)
    try:
        ax.set_box_aspect(1.0)
    except AttributeError:
        pass
    if equal_axis:
        ax.set_aspect("equal", adjustable="box")


def is_three_dimensional_OCPSolution(*arrays: np.ndarray, tol: float = 1e-8) -> bool:
    return any(array is not None and np.max(np.abs(array[2])) > tol for array in arrays)


def plot_system_points_2d(ax, case: TestCase, axis_0: int, axis_1: int, lagrange: dict[str, float]) -> None:
    moon_coord = np.zeros(3)
    moon_coord[0] = 1.0 - case.mu
    ax.scatter(
        moon_coord[axis_0],
        moon_coord[axis_1],
        color=CALEB_MOON_COLOR,
        edgecolor=DARK_GREY_COLOR,
        linewidth=AXIS_SPINE_WIDTH,
        marker="o",
        s=70,
        zorder=6,
    )
    ax.text(moon_coord[axis_0], moon_coord[axis_1], " MOON", fontsize=ANNOTATION_FONT_SIZE)
    for name in ("L1", "L2"):
        point = np.zeros(3)
        point[0] = lagrange[name]
        ax.scatter(point[axis_0], point[axis_1], color=DARK_GREY_COLOR, marker="o", s=16, zorder=6)
        ax.text(point[axis_0], point[axis_1], f" ${name[0]}_{name[1]}$", fontsize=ANNOTATION_FONT_SIZE)


def plot_projection(
    ax,
    case: TestCase,
    x_dense: np.ndarray,
    u_dense: np.ndarray,
    departure_orbit: np.ndarray | None,
    target_orbit: np.ndarray | None,
    lagrange: dict[str, float],
    axis_0: int,
    axis_1: int,
    trajectory_color: str | None = None,
) -> None:
    trajectory_color = trajectory_color or PROJECTION_COLORS.get((axis_0, axis_1), CALEB_TRAJECTORY_COLOR)
    if departure_orbit is not None:
        ax.plot(
            departure_orbit[axis_0],
            departure_orbit[axis_1],
            color=CALEB_REFERENCE_COLOR,
            linestyle="dashed",
            lw=REFERENCE_LINE_WIDTH,
        )
    if target_orbit is not None:
        ax.plot(
            target_orbit[axis_0],
            target_orbit[axis_1],
            color=CALEB_REFERENCE_COLOR,
            linestyle="dotted",
            lw=REFERENCE_LINE_WIDTH,
        )

    ax.plot(x_dense[axis_0], x_dense[axis_1], color=trajectory_color, lw=TRAJECTORY_LINE_WIDTH, zorder=4)
    ax.scatter(x_dense[axis_0, 0], x_dense[axis_1, 0], color=CALEB_POINT_COLOR, marker="^", s=28, zorder=7)
    ax.text(x_dense[axis_0, 0], x_dense[axis_1, 0], " $x_0$", fontsize=ANNOTATION_FONT_SIZE)
    ax.scatter(x_dense[axis_0, -1], x_dense[axis_1, -1], color=CALEB_POINT_COLOR, marker="v", s=28, zorder=7)
    ax.text(x_dense[axis_0, -1], x_dense[axis_1, -1], " $x_t$", fontsize=ANNOTATION_FONT_SIZE)

    u_norm = np.linalg.norm(u_dense, axis=0)
    active = np.flatnonzero(u_norm > 0.03 * case.max_thrust_nd)
    if active.size:
        chosen = active[np.unique(np.linspace(0, active.size - 1, min(36, active.size)).astype(int))]
        direction = u_dense[:, chosen] / np.maximum(u_norm[chosen], 1e-14)
        thrust_ratio = u_norm[chosen] / max(case.max_thrust_nd, float(np.max(u_norm)))
        arrow_scale_lu = 0.035
        ax.quiver(
            x_dense[axis_0, chosen],
            x_dense[axis_1, chosen],
            direction[axis_0] * thrust_ratio * arrow_scale_lu,
            direction[axis_1] * thrust_ratio * arrow_scale_lu,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=CALEB_THRUST_COLOR,
            width=0.0045,
            headwidth=3.5,
            headlength=4.5,
            alpha=0.85,
            zorder=5,
        )

    plot_system_points_2d(ax, case, axis_0, axis_1, lagrange)
    ax.set_xlabel(AXIS_LABELS[axis_0])
    ax.set_ylabel(AXIS_LABELS[axis_1])
    style_2d_axis(ax, equal_axis=True)


def save_projection_plot(
    case: TestCase,
    x_dense: np.ndarray,
    u_dense: np.ndarray,
    departure_orbit: np.ndarray | None,
    target_orbit: np.ndarray | None,
    lagrange: dict[str, float],
    axis_0: int,
    axis_1: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE, dpi=FIGURE_DPI, constrained_layout=True)
    plot_projection(
        ax,
        case,
        x_dense,
        u_dense,
        departure_orbit,
        target_orbit,
        lagrange,
        axis_0,
        axis_1,
        trajectory_color=PROJECTION_COLORS[(axis_0, axis_1)],
    )
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def set_axes_equal_3d(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    ranges = np.array([x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0]])
    centers = np.array([sum(x_limits), sum(y_limits), sum(z_limits)]) * 0.5
    radius = 0.5 * max(ranges)
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)


def save_3d_plot(
    case: TestCase,
    x_dense: np.ndarray,
    u_dense: np.ndarray,
    departure_orbit: np.ndarray | None,
    target_orbit: np.ndarray | None,
    lagrange: dict[str, float],
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=THREE_D_FIGSIZE, dpi=FIGURE_DPI)
    ax = fig.add_subplot(111, projection="3d")
    if departure_orbit is not None:
        ax.plot(
            departure_orbit[0],
            departure_orbit[1],
            departure_orbit[2],
            color=CALEB_REFERENCE_COLOR,
            linestyle="dashed",
            lw=REFERENCE_LINE_WIDTH,
        )
    if target_orbit is not None:
        ax.plot(
            target_orbit[0],
            target_orbit[1],
            target_orbit[2],
            color=CALEB_REFERENCE_COLOR,
            linestyle="dotted",
            lw=REFERENCE_LINE_WIDTH,
        )
    ax.plot(x_dense[0], x_dense[1], x_dense[2], color=CALEB_TRAJECTORY_COLOR, lw=TRAJECTORY_LINE_WIDTH)
    ax.scatter(x_dense[0, 0], x_dense[1, 0], x_dense[2, 0], color=CALEB_POINT_COLOR, marker="^", s=26)
    ax.scatter(x_dense[0, -1], x_dense[1, -1], x_dense[2, -1], color=CALEB_POINT_COLOR, marker="v", s=26)
    ax.text(x_dense[0, 0], x_dense[1, 0], x_dense[2, 0], " $x_0$", fontsize=ANNOTATION_FONT_SIZE)
    ax.text(x_dense[0, -1], x_dense[1, -1], x_dense[2, -1], " $x_t$", fontsize=ANNOTATION_FONT_SIZE)
    moon = np.array([1.0 - case.mu, 0.0, 0.0])
    ax.scatter(
        moon[0],
        moon[1],
        moon[2],
        color=CALEB_MOON_COLOR,
        edgecolor=DARK_GREY_COLOR,
        linewidth=AXIS_SPINE_WIDTH,
        marker="o",
        s=70,
    )
    ax.text(moon[0], moon[1], moon[2], " MOON", fontsize=ANNOTATION_FONT_SIZE)
    for name in ("L1", "L2"):
        ax.scatter(lagrange[name], 0.0, 0.0, color=DARK_GREY_COLOR, marker="o", s=16)
        ax.text(lagrange[name], 0.0, 0.0, f" ${name[0]}_{name[1]}$", fontsize=ANNOTATION_FONT_SIZE)
    ax.set_xlabel("x [LU]")
    ax.set_ylabel("y [LU]")
    ax.set_zlabel("z [LU]")
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.zaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE, width=MAJOR_TICK_WIDTH)
    ax.view_init(azim=78, elev=15)
    ax.grid(True, color=LIGHT_GREY_COLOR, alpha=0.7)
    set_axes_equal_3d(ax)
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def moon_surface_distance_km(case: TestCase, x_values: np.ndarray) -> np.ndarray:
    moon_relative_position = x_values[0:3] - np.array([[1.0 - case.mu], [0.0], [0.0]])
    moon_center_distance_lu = np.linalg.norm(moon_relative_position, axis=0)
    return (moon_center_distance_lu - case.moon_radius_lu) * case.length_unit


def save_moon_distance_plot(
    case: TestCase,
    t_dense_days: np.ndarray,
    moon_distance_km: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE, dpi=FIGURE_DPI, constrained_layout=True)
    ax.plot(t_dense_days, moon_distance_km, color=SECONDARY_DATA_COLOR, lw=TRAJECTORY_LINE_WIDTH)
    ax.axhline(0.0, color=CALEB_THRUST_COLOR, linestyle="dotted", lw=GUIDE_LINE_WIDTH)
    closest_idx = int(np.argmin(moon_distance_km))
    ax.scatter(t_dense_days[closest_idx], moon_distance_km[closest_idx], color=POINT_MARKER_COLOR, s=22, zorder=4)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("distance to Moon surface [km]")
    style_2d_axis(ax)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_verification_plots(
    t_dense_days: np.ndarray,
    position_error_m: np.ndarray,
    velocity_error_m_per_s: np.ndarray,
    mass_consumption_abs_difference_kg: np.ndarray,
    output_prefix: Path,
) -> list[Path]:
    plot_error_m = np.maximum(position_error_m, 1e-16)
    plot_velocity_error_m_per_s = np.maximum(velocity_error_m_per_s, 1e-16)
    plot_mass_error_kg = np.maximum(mass_consumption_abs_difference_kg, 1e-16)
    specs = [
        ("verification_position", "absolute position difference [m]", plot_error_m, position_error_m, TRAJECTORY_COLOR),
        ("verification_velocity", "absolute velocity difference [m/s]", plot_velocity_error_m_per_s, velocity_error_m_per_s, SECONDARY_DATA_COLOR),
        ("verification_mass", "mass-consumption difference [kg]", plot_mass_error_kg, mass_consumption_abs_difference_kg, THRUST_COLOR),
    ]
    saved_paths: list[Path] = []
    for suffix, ylabel, y_plot, y_raw, color in specs:
        fig, ax = plt.subplots(figsize=WIDE_FIGSIZE, dpi=FIGURE_DPI, constrained_layout=True)
        ax.semilogy(t_dense_days, y_plot, color=color, lw=TRAJECTORY_LINE_WIDTH)
        max_idx = int(np.argmax(y_raw))
        ax.scatter(t_dense_days[max_idx], max(y_raw[max_idx], 1e-16), color=CALEB_THRUST_COLOR, s=22, zorder=4)
        ax.set_xlabel("time [days]")
        ax.set_ylabel(ylabel)
        style_2d_axis(ax, grid_which="both")
        output_path = output_prefix.with_name(f"{output_prefix.name}_{suffix}").with_suffix(".png")
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)
    return saved_paths


def save_thrust_plot(
    t_days: np.ndarray,
    u_norm_n: np.ndarray,
    max_thrust_n: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=WIDE_FIGSIZE, dpi=FIGURE_DPI, constrained_layout=True)
    ax.step(t_days, u_norm_n, where="post", color=CALEB_THRUST_COLOR, lw=TRAJECTORY_LINE_WIDTH, label="Thrust")
    ax.axhline(max_thrust_n, color=LIMIT_COLOR, linestyle="dotted", lw=GUIDE_LINE_WIDTH, label="Max thrust")
    ax.set_xlabel("time [days]")
    ax.set_ylabel("thrust [N]")
    ax.legend(loc="best", fontsize=LEGEND_FONT_SIZE, frameon=True, edgecolor=GREY_COLOR)
    style_2d_axis(ax)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    case: TestCase,
    OCPSolution: OCPSolution,
    history: list[dict[str, float]],
    defects: dict[str, np.ndarray],
    options: HPAdaptiveOptions,
    output_prefix: Path,
) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    t_days = OCPSolution.mesh * case.tof_days
    u_norm_n = np.linalg.norm(OCPSolution.u, axis=0) * case.thrust_unit
    t_dense_days, x_dense, u_dense, sigma_dense = interpolate_hp_OCPSolution(case, OCPSolution)
    u_dense_norm_n = np.linalg.norm(u_dense, axis=0) * case.thrust_unit
    moon_distance_surface_km = moon_surface_distance_km(case, x_dense)
    (
        x_integrated,
        position_error_lu,
        position_error_m,
        velocity_error_nd,
        velocity_error_m_per_s,
        mass_consumption_difference_kg,
        mass_consumption_abs_difference_kg,
    ) = compute_augm_state_error(
        case,
        t_dense_days,
        x_dense,
        u_dense,
        sigma_dense,
    )

    stage_interval: list[int] = []
    stage_degree_index: list[int] = []
    stage_tau: list[float] = []
    stage_state: list[np.ndarray] = []
    for (interval, j), state in sorted(OCPSolution.stages.items()):
        tau_root, _, _, _ = collocation_coefficients(int(OCPSolution.degrees[interval]))
        stage_interval.append(interval)
        stage_degree_index.append(j)
        stage_tau.append(float(tau_root[j]))
        stage_state.append(state)

    npz_path = output_prefix.with_suffix(".npz")
    np.savez(
        npz_path,
        mesh_fraction=OCPSolution.mesh,
        interval_degrees=OCPSolution.degrees,
        t_days=t_days,
        x=OCPSolution.x,
        u=OCPSolution.u,
        sigma=OCPSolution.sigma,
        u_norm_n=u_norm_n,
        stage_interval=np.asarray(stage_interval, dtype=int),
        stage_degree_index=np.asarray(stage_degree_index, dtype=int),
        stage_tau=np.asarray(stage_tau, dtype=float),
        stage_state=np.asarray(stage_state, dtype=float),
        t_dense_days=t_dense_days,
        x_dense=x_dense,
        u_dense=u_dense,
        sigma_dense=sigma_dense,
        u_dense_norm_n=u_dense_norm_n,
        moon_distance_surface_km=moon_distance_surface_km,
        x_integrated=x_integrated,
        position_error_lu=position_error_lu,
        position_error_m=position_error_m,
        velocity_error_nd=velocity_error_nd,
        velocity_error_m_per_s=velocity_error_m_per_s,
        mass_consumption_difference_kg=mass_consumption_difference_kg,
        mass_consumption_abs_difference_kg=mass_consumption_abs_difference_kg,
        interval_defect_scaled=defects["scaled"],
        interval_defect_position_m=defects["position_m"],
        interval_defect_velocity_m_per_s=defects["velocity_m_per_s"],
        interval_defect_mass_kg=defects["mass_kg"],
        history=np.array(history, dtype=object),
    )

    header = "t_days,x,y,z,vx,vy,vz,m_nd,ux_N,uy_N,uz_N,sigma_N,thrust_norm_N"
    dense_csv = np.column_stack(
        [
            t_dense_days,
            x_dense.T,
            u_dense.T * case.thrust_unit,
            sigma_dense.T * case.thrust_unit,
            u_dense_norm_n,
        ]
    )
    csv_path = output_prefix.with_suffix(".csv")
    np.savetxt(csv_path, dense_csv, delimiter=",", header=header, comments="")

    lagrange = collinear_lagrange_points(case)
    departure_orbit = None
    target_orbit = None
    if case.departure_period_nd is not None:
        departure_orbit = propagate_periodic_orbit(case, case.x0_augmented_state, case.departure_period_nd)
    if case.target_period_nd is not None:
        target_orbit = propagate_periodic_orbit(case, case.xf_augmented_state, case.target_period_nd)

    saved_paths = [npz_path, csv_path]
    projection_axes = (
        LYAPUNOV_L1_TO_L2_PROJECTION_AXES
        if case.test_case_id == "lyapunov_l1_to_l2"
        else PROJECTION_AXES
    )
    for axis_0, axis_1 in projection_axes:
        suffix = PROJECTION_SUFFIXES[(axis_0, axis_1)]
        projection_path = output_prefix.with_name(f"{output_prefix.name}_{suffix}").with_suffix(".png")
        save_projection_plot(
            case,
            x_dense,
            u_dense,
            departure_orbit,
            target_orbit,
            lagrange,
            axis_0,
            axis_1,
            projection_path,
        )
        saved_paths.append(projection_path)

    thrust_path = output_prefix.with_name(f"{output_prefix.name}_thrust").with_suffix(".png")
    save_thrust_plot(t_days, u_norm_n, case.max_thrust_n, thrust_path)
    saved_paths.append(thrust_path)
    moon_distance_path = output_prefix.with_name(f"{output_prefix.name}_moon_distance").with_suffix(".png")
    save_moon_distance_plot(case, t_dense_days, moon_distance_surface_km, moon_distance_path)
    saved_paths.append(moon_distance_path)
    saved_paths.extend(save_verification_plots(
        t_dense_days,
        position_error_m,
        velocity_error_m_per_s,
        mass_consumption_abs_difference_kg,
        output_prefix,
    ))
    figure_3d_path = output_prefix.with_name(f"{output_prefix.name}_3d").with_suffix(".png")
    save_3d_plot(case, x_dense, u_dense, departure_orbit, target_orbit, lagrange, figure_3d_path)
    saved_paths.append(figure_3d_path)
    saved_paths.extend(save_defect_plots(case, OCPSolution, defects, options, output_prefix))
    return saved_paths


def save_defect_plots(
    case: TestCase,
    OCPSolution: OCPSolution,
    defects: dict[str, np.ndarray],
    options: HPAdaptiveOptions,
    output_prefix: Path,
) -> list[Path]:
    centers = 0.5 * (OCPSolution.mesh[:-1] + OCPSolution.mesh[1:]) * case.tof_days
    widths = np.diff(OCPSolution.mesh) * case.tof_days
    scaled = np.maximum(defects["scaled"], 1e-16)

    saved_paths: list[Path] = []
    defect_path = output_prefix.with_name(f"{output_prefix.name}_defect").with_suffix(".png")
    fig, defect_ax = plt.subplots(figsize=WIDE_FIGSIZE, dpi=FIGURE_DPI, constrained_layout=True)
    defect_ax.bar(centers, scaled, width=widths, align="center", color=CALEB_TRAJECTORY_COLOR, alpha=0.85)
    defect_ax.axhline(options.defect_tolerance, color=CALEB_THRUST_COLOR, linestyle="dotted", lw=GUIDE_LINE_WIDTH)
    defect_ax.set_yscale("log")
    defect_ax.set_xlabel("time [days]")
    defect_ax.set_ylabel("scaled endpoint defect")
    style_2d_axis(defect_ax, grid_which="both")
    fig.savefig(defect_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(defect_path)

    degree_path = output_prefix.with_name(f"{output_prefix.name}_degree").with_suffix(".png")
    fig, degree_ax = plt.subplots(figsize=WIDE_FIGSIZE, dpi=FIGURE_DPI, constrained_layout=True)
    degree_ax.step(
        OCPSolution.mesh * case.tof_days,
        np.r_[OCPSolution.degrees, OCPSolution.degrees[-1]],
        where="post",
        color=SECONDARY_DATA_COLOR,
        lw=TRAJECTORY_LINE_WIDTH,
    )
    degree_ax.set_xlabel("time [days]")
    degree_ax.set_ylabel("Radau degree")
    degree_ax.set_ylim(options.min_degree - 0.5, options.max_degree + 0.5)
    style_2d_axis(degree_ax)
    fig.savefig(degree_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(degree_path)
    return saved_paths


def run_hp_adaptive(
    case: TestCase,
    options: HPAdaptiveOptions,
    max_iter: int,
    print_level: int,
    log_prefix: str = "",
) -> tuple[OCPSolution, list[dict[str, float]], dict[str, np.ndarray]]:
    mesh = np.linspace(0.0, 1.0, options.initial_intervals + 1)
    degrees = np.full(options.initial_intervals, options.min_degree, dtype=int)
    guess: INITIAL_GUESS = None
    history: list[dict[str, float]] = []
    prefer_hs_warm_start = False
    OCPSolution: OCPSolution | None = None
    defects: dict[str, np.ndarray] | None = None

    print(
        f"\n{log_prefix}Hermite-Simpson pre-warm: nodes={options.initial_intervals}",
        flush=True,
    )
    hs_x, hs_u, hs_sigma, hs_diag = _warm_start(
        case=case,
        nodes=options.initial_intervals,
        sol=None,
        max_iter=max_iter,
        print_level=print_level,
    )
    print(
        log_prefix
        + "  pre-warm objective={fuel_consumed_kg:.6f} kg, "
        "terminal_error={terminal_error_nd:.3e}".format(**hs_diag),
        flush=True,
    )
    guess = (hs_x, hs_u, hs_sigma)

    for adaptation_index in range(1, options.max_adapt_iterations + 1):
        print(
            f"\n{log_prefix}HP iteration {adaptation_index}: "
            f"intervals={degrees.size}, p=[{np.min(degrees)}, {np.max(degrees)}]",
            flush=True,
        )
        OCPSolution, used_hs_warm_start = solve_hp_with_retries(
            case=case,
            mesh=mesh,
            degrees=degrees,
            guess=guess,
            options=options,
            max_iter=max_iter,
            print_level=print_level,
            log_prefix=log_prefix,
            prefer_hermite_simpson_warm_start=prefer_hs_warm_start,
        )
        defects = estimate_interval_defects(case, OCPSolution, options.rk4_substeps_per_interval)
        max_defect = float(np.max(defects["scaled"]))
        mean_defect = float(np.mean(defects["scaled"]))
        max_defect_idx = int(np.argmax(defects["scaled"]))

        diag = dict(OCPSolution.diagnostics)
        diag["adapt_iteration"] = float(adaptation_index)
        diag["max_interval_defect"] = max_defect
        diag["mean_interval_defect"] = mean_defect
        diag["max_defect_interval"] = float(max_defect_idx)
        diag["used_hs_warm_start"] = float(used_hs_warm_start)
        history.append(diag)

        print(
            log_prefix
            + "  objective={fuel_consumed_kg:.6f} kg, intervals={intervals:.0f}, "
            "p=[{min_degree:.0f}, {max_degree:.0f}], max_defect={max_interval_defect:.3e}, "
            "terminal_error={terminal_error_nd:.3e}".format(**diag),
            flush=True,
        )

        if max_defect <= options.defect_tolerance:
            print(f"{log_prefix}  hp tolerance reached.", flush=True)
            break

        new_mesh, new_degrees, refinement = refine_hp_mesh(OCPSolution, defects["scaled"], options)
        print(
            f"{log_prefix}  refine: split={refinement['split_count']:.0f}, "
            f"p+={refinement['p_increase_count']:.0f}, p-={refinement['p_decrease_count']:.0f}",
            flush=True,
        )
        if refinement["changed_count"] <= 0.0:
            print(f"{log_prefix}  hp refinement limit reached.", flush=True)
            break

        guess = OCPSolution
        mesh = new_mesh
        degrees = new_degrees
        prefer_hs_warm_start = used_hs_warm_start and degrees.size != OCPSolution.degrees.size

    assert OCPSolution is not None
    assert defects is not None
    return OCPSolution, history, defects


def output_prefix_for_case(output_dir: Path, output_prefix: str, case: TestCase) -> Path:
    return output_dir / case.test_case_id / output_prefix


def run_case(test_case_id: str) -> tuple[str, str, list[str]]:
    case = CASE_REGISTRY[test_case_id]()
    options = HPAdaptiveOptions()
    output_prefix = output_prefix_for_case(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_PREFIX, case)
    log_prefix = f"[{case.test_case_id}] "

    print(f"\n{log_prefix}Starting {case.display_name} with hp-adaptive Radau", flush=True)
    print(
        f"{log_prefix}initial_intervals={options.initial_intervals}, "
        f"max_intervals={options.max_intervals}, p=[{options.min_degree}, {options.max_degree}], "
        f"defect_tol={options.defect_tolerance:g}, ",
        flush=True,
    )

    OCPSolution, history, defects = run_hp_adaptive(
        case=case,
        options=options,
        max_iter=MAX_ITER,
        print_level=PRINT_LEVEL,
        log_prefix=log_prefix,
    )
    saved_paths = save_outputs(case, OCPSolution, history, defects, options, output_prefix)

    print(f"\n{log_prefix}Saved:", flush=True)
    for path in saved_paths:
        print(f"{log_prefix}  {path}", flush=True)

    return case.test_case_id, case.display_name, [str(path) for path in saved_paths]


def main() -> None:
    process_count = min(len(CASE_REGISTRY), mp.cpu_count())
    print(
        f"Running {len(CASE_REGISTRY)} test cases in parallel "
        f"with {process_count} processes.",
        flush=True,
    )
    print(f"Output directory: {DEFAULT_OUTPUT_DIR.resolve()}", flush=True)

    context = mp.get_context("spawn")
    results: dict[str, tuple[str, list[str]]] = {}
    with ProcessPoolExecutor(max_workers=process_count, mp_context=context) as executor:
        futures = {executor.submit(run_case, test_case_id): test_case_id for test_case_id in CASE_REGISTRY}
        for future in as_completed(futures):
            test_case_id = futures[future]
            try:
                finished_case_id, display_name, saved_paths = future.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"Case '{test_case_id}' failed.") from exc
            results[finished_case_id] = (display_name, saved_paths)
            print(f"\n[{finished_case_id}] Finished.", flush=True)

    print("\nAll test cases finished.", flush=True)
    for test_case_id in CASE_REGISTRY:
        display_name, saved_paths = results[test_case_id]
        print(f"\n{display_name}", flush=True)
        for path in saved_paths:
            print(f"  {path}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
