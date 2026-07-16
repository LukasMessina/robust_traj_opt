"""
Compute Earth-Moon CR3BP low-thrust transfers with h-adaptive Radau collocation.

The continuous-time optimal control problem is discretized with a variable-size,
fixed-degree Radau collocation mesh, formulated with CasADi, and solved with IPOPT.
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import multiprocessing as mp
import casadi
import numpy as np
import integrator
from plotter import Plotter

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
    departure_period: float | None = None       # [-]
    target_period: float | None = None          # [-]
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
    departure_period: float = 2.9750964922007723  # [-]
    target_period: float = 3.49306635929003       # [-]


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
    departure_period: float = 3.2746644337639852  # [-]
    target_period: float = 2.5748200748171399     # [-]


@dataclass
class OCPSolution:
    mesh: np.ndarray
    x: np.ndarray
    u: np.ndarray
    sigma: np.ndarray
    diagnostics: dict[str, float]
    degrees: np.ndarray | None = None
    stages: dict[tuple[int, int], np.ndarray] | None = None

    @property
    def is_hp_radau(self) -> bool:
        return self.degrees is not None and self.stages is not None

    @property
    def is_hermite_simpson(self) -> bool:
        return self.degrees is None and self.stages is None


@dataclass(frozen=True)
class HAdaptiveOptions:
    initial_intervals: int = 150
    radau_degree: int = 3
    max_intervals: int = 1000
    max_adapt_iterations: int = 20
    defect_tolerance: float = 1e-11
    rk4_substeps_per_interval: int = 32
    

CASE_TYPES: tuple[type[TestCase], ...] = (LyapunovL1ToL2, HaloL2ToHaloL1)
CASE_REGISTRY: dict[str, type[TestCase]] = {
    case_type().test_case_id: case_type for case_type in CASE_TYPES
}
MAX_ITER = 10000
PRINT_LEVEL = 0
TOL = 1e-9
DEFAULT_OUTPUT_DIR = Path("output/cr3bp")
DEFAULT_OUTPUT_PREFIX = ""
INITIAL_GUESS = OCPSolution | None
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

def get_collinear_lagrange_points(case: CR3BPEarthMoon) -> dict[str, float]:
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

def _resample_hs_solution(
    sol: OCPSolution | None,
    nodes: int,
    case: TestCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sol is None:
        return _rollout(case, nodes)

    new_grid = np.linspace(0.0, 1.0, nodes + 1)
    x_guess = np.column_stack([evaluate_solution_state(case, sol, fraction) for fraction in new_grid])
    u_guess = np.vstack([np.interp(new_grid, sol.mesh, sol.u[i]) for i in range(sol.u.shape[0])])
    sigma_guess = np.vstack([np.interp(new_grid, sol.mesh, sol.sigma[0])])
    return x_guess, u_guess, sigma_guess

def _warm_start(
    case: TestCase,
    nodes: int,
    sol: OCPSolution | None,
    max_iter: int,
    print_level: int,
    extra_options: dict[str, float | int | str] | None = None,
) -> OCPSolution:
    "Solve the optimal control problem with a Hermite-Simpson collocation method to warm-start the h-adaptive Radau refinement."
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

    x_guess, u_guess, sigma_guess = _resample_hs_solution(sol, nodes, case)
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
    opti.solver("ipopt", {"expand": True, "print_time": False}, ipopt_settings)

    sol = opti.solve()
    x_sol = np.asarray(sol.value(x_var), dtype=float)
    u_sol = np.asarray(sol.value(u_var), dtype=float)
    sigma_sol = np.asarray(sol.value(sigma_var), dtype=float).reshape(1, -1)
    diagnostics = get_diagnostics(case, x_sol, u_sol, sigma_sol)
    diagnostics["fuel_consumed_kg"] = float(sol.value(fuel_consumed))
    diagnostics["nodes"] = float(nodes)
    return OCPSolution(
        mesh=np.linspace(0.0, 1.0, nodes + 1),
        x=x_sol,
        u=u_sol,
        sigma=sigma_sol,
        diagnostics=diagnostics,
    )

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

def rollout_initial_guess(
    case: TestCase,
    mesh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intervals = mesh.size - 1
    u_component_nd = 1e-6 / case.thrust_unit
    u_guess_nd = np.full((3, intervals + 1), u_component_nd, dtype=float)
    sigma_guess_nd = np.full((1, intervals + 1), np.linalg.norm(u_guess_nd[:, 0]), dtype=float)

    x_guess = np.empty((7, intervals + 1), dtype=float)
    x_guess[:, 0] = case.x0_augmented_state
    for k in range(intervals):
        h = case.tof_nd * float(mesh[k + 1] - mesh[k])
        x_guess[:, k + 1] = integrator.rk4(
            case,
            x_guess[:, k],
            u_guess_nd[:, k],
            sigma_guess_nd[0, k],
            h,
        )

    return x_guess, u_guess_nd, sigma_guess_nd


def evaluate_hp_state(solution: OCPSolution, fraction: float) -> np.ndarray:

    interval = int(np.searchsorted(solution.mesh, fraction, side="right") - 1)
    interval = min(max(interval, 0), solution.degrees.size - 1)
    left = float(solution.mesh[interval])
    right = float(solution.mesh[interval + 1])
    theta = (fraction - left) / (right - left)
    degree = int(solution.degrees[interval])
    tau_root, _, _, _ = collocation_coefficients(degree)
    interval_states = [solution.x[:, interval]]
    for j in range(1, degree):
        interval_states.append(solution.stages[(interval, j)])
    interval_states.append(solution.x[:, interval + 1])

    basis = lagrange_basis_values(theta, tau_root)
    state = np.zeros(7, dtype=float)
    for coefficient, interval_state in zip(basis, interval_states):
        state += coefficient * interval_state
    return state


def evaluate_hs_state(case: TestCase, solution: OCPSolution, fraction: float) -> np.ndarray:

    interval = int(np.searchsorted(solution.mesh, fraction, side="right") - 1)
    interval = min(max(interval, 0), solution.mesh.size - 2)
    left = float(solution.mesh[interval])
    right = float(solution.mesh[interval + 1])
    theta = (fraction - left) / (right - left)
    h = case.tof_nd * (right - left)
    tau = theta * h

    x_k = solution.x[:, interval]
    x_kp1 = solution.x[:, interval + 1]
    u_k = solution.u[:, interval]
    u_kp1 = solution.u[:, interval + 1]
    sigma_k = float(solution.sigma[0, interval])
    sigma_kp1 = float(solution.sigma[0, interval + 1])

    xdot_k = eom(case, x_k, u_k, sigma_k)
    xdot_kp1 = eom(case, x_kp1, u_kp1, sigma_kp1)
    x_c = 0.5 * (x_k + x_kp1) + h / 8.0 * (xdot_k - xdot_kp1)
    u_c = 0.5 * (u_k + u_kp1)
    sigma_c = 0.5 * (sigma_k + sigma_kp1)
    xdot_c = eom(case, x_c, u_c, sigma_c)

    tau2 = tau * tau
    tau3 = tau2 * tau
    return (
        x_k
        + tau * xdot_k
        - tau2 / (2.0 * h) * (3.0 * xdot_k - 4.0 * xdot_c + xdot_kp1)
        + tau3 / (3.0 * h * h) * (2.0 * xdot_k - 4.0 * xdot_c + 2.0 * xdot_kp1)
    )


def evaluate_solution_state(case: TestCase, solution: OCPSolution, fraction: float) -> np.ndarray:
    if solution.is_hp_radau:
        return evaluate_hp_state(solution, fraction)
    return evaluate_hs_state(case, solution, fraction)


def sample_endpoint_variables(
    case: TestCase,
    guess: INITIAL_GUESS,
    mesh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if guess is None:
        return rollout_initial_guess(case, mesh)

    x_guess = np.column_stack([evaluate_solution_state(case, guess, fraction) for fraction in mesh])
    u_guess = np.vstack([np.interp(mesh, guess.mesh, guess.u[i]) for i in range(guess.u.shape[0])])
    sigma_guess = np.vstack([np.interp(mesh, guess.mesh, guess.sigma[0])])
    return x_guess, u_guess, sigma_guess


def solve_ocp(
    case: TestCase,
    mesh: np.ndarray,
    degrees: np.ndarray,
    guess: INITIAL_GUESS,
    max_iter: int,
    print_level: int,
    log_prefix: str,
) -> OCPSolution:

    attempts = [
        ("adaptive", {"mu_strategy": "adaptive"}),
        ("monotone", {"mu_strategy": "monotone"}),
    ]
    last_error: Exception | None = None
    for label, extra_options in attempts:
        try:
            print(f"{log_prefix}  Barrier parameter strategy: {label}", flush=True)
            opti = casadi.Opti()
            intervals = degrees.size
            max_thrust_nd = case.max_thrust_nd
            x_var = opti.variable(7, intervals + 1)
            u_var = opti.variable(3, intervals + 1)
            sigma_var = opti.variable(1, intervals + 1)
            stage_vars: list[tuple[int, int, casadi.MX]] = []
            fuel_consumed_nd = 0.0

            for k in range(intervals + 1):
                opti.subject_to(casadi.dot(u_var[:, k], u_var[:, k]) <= sigma_var[0, k] ** 2)
                opti.subject_to(opti.bounded(0.0, sigma_var[0, k], max_thrust_nd))
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


                fuel_consumed_nd += h * b_vector[0] * sigma_var[0, k] / case.exhaust_velocity_nd

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

            x_guess, u_guess, sigma_guess = sample_endpoint_variables(case, guess, mesh)
            sigma_guess = np.maximum(sigma_guess, 1e-7)
            opti.set_initial(x_var, x_guess)
            opti.set_initial(u_var, u_guess)
            opti.set_initial(sigma_var, sigma_guess)

            for k, j, x_stage in stage_vars:
                degree = int(degrees[k])
                tau_root, _, _, _ = collocation_coefficients(degree)
                fraction = float(mesh[k] + tau_root[j] * (mesh[k + 1] - mesh[k]))
                stage_guess = evaluate_solution_state(case, guess, fraction)
                opti.set_initial(x_stage, stage_guess)

            ipopt_options: dict[str, float | int | str] = {
                "max_iter": max_iter,
                "tol": TOL,
                "acceptable_tol": TOL,
                "constr_viol_tol": TOL,
                "dual_inf_tol": TOL,
                "compl_inf_tol": TOL,
                "print_level": print_level,
                "sb": "yes",
            }

            ipopt_options.update(extra_options)

            opti.solver("ipopt", {"expand": True, "print_time": False}, ipopt_options)

            sol = opti.solve()
            x_sol = np.asarray(sol.value(x_var), dtype=float)
            u_sol = np.asarray(sol.value(u_var), dtype=float)
            sigma_sol = np.asarray(sol.value(sigma_var), dtype=float).reshape(1, -1)
            stages_sol = {
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
                stages=stages_sol,
                diagnostics=diagnostics,
            )
        except RuntimeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error


def estimate_interval_defects(
    case: TestCase,
    OCPSolution: OCPSolution,
    substeps: int,
) -> dict[str, np.ndarray]:
    intervals = OCPSolution.degrees.size
    substep_grid = np.arange(substeps + 1, dtype=float) / substeps
    scaled_error = np.empty(intervals, dtype=float)
    position_error = np.empty(intervals, dtype=float)              # [m]
    velocity_error = np.empty(intervals, dtype=float)              # [m/s]
    mass_error = np.empty(intervals, dtype=float)                  # [kg]

    for k in range(intervals):
        h = case.tof_nd * float(OCPSolution.mesh[k + 1] - OCPSolution.mesh[k])
        dt = h / substeps
        u0 = OCPSolution.u[:, k]
        u1 = OCPSolution.u[:, k + 1]
        sigma0 = float(OCPSolution.sigma[0, k])
        sigma1 = float(OCPSolution.sigma[0, k + 1])
        x_integrated = OCPSolution.x[:, k].copy()

        for tau_a, tau_b in zip(substep_grid[:-1], substep_grid[1:]):
            x_integrated = integrator.rk4(
                case,
                x_integrated,
                (1.0 - tau_a) * u0 + tau_a * u1,
                (1.0 - tau_a) * sigma0 + tau_a * sigma1,
                dt,
                control_end=(1.0 - tau_b) * u0 + tau_b * u1,
                sigma_end=(1.0 - tau_b) * sigma0 + tau_b * sigma1,
            )

        error = x_integrated - OCPSolution.x[:, k + 1]
        scale = np.maximum(1.0, np.maximum(np.abs(OCPSolution.x[:, k]), np.abs(OCPSolution.x[:, k + 1])))
        scaled_error[k] = float(np.max(np.abs(error) / scale))
        position_error[k] = float(np.linalg.norm(error[0:3]) * case.length_unit * 1000.0)         # [m]
        velocity_error[k] = float(np.linalg.norm(error[3:6]) * case.velocity_unit * 1000.0)       # [m/s]   
        mass_error[k] = float(abs(error[6]) * case.m0_wet)                                        # [kg]

    return {
        "scaled": scaled_error,
        "position": position_error,
        "velocity": velocity_error,
        "mass_kg": mass_error,
    }


def refine_h_mesh(
    OCPSolution: OCPSolution,
    defects: np.ndarray,
    options: HAdaptiveOptions,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mesh = OCPSolution.mesh
    degrees = OCPSolution.degrees
    intervals = degrees.size
    over_tol = defects > options.defect_tolerance

    remaining_capacity = max(options.max_intervals - intervals, 0)
    split_selected = set(
        sorted(np.flatnonzero(over_tol), key=lambda idx: defects[idx], reverse=True)[:remaining_capacity]
    )

    new_mesh = [float(mesh[0])]
    new_degrees: list[int] = []

    for k in range(intervals):
        degree = int(degrees[k])
        if k in split_selected:
            midpoint = 0.5 * float(mesh[k] + mesh[k + 1])
            new_degrees.append(degree)
            new_mesh.append(midpoint)
            new_degrees.append(degree)
            new_mesh.append(float(mesh[k + 1]))
            continue

        new_degrees.append(degree)
        new_mesh.append(float(mesh[k + 1]))

    new_mesh_array = np.asarray(new_mesh, dtype=float)
    new_degrees_array = np.asarray(new_degrees, dtype=int)
    summary = {
        "split_count": float(len(split_selected)),
        "changed_count": float(len(split_selected)),
    }
    return new_mesh_array, new_degrees_array, summary


def interpolate_OCPSolution(
    case: TestCase,
    OCPSolution: OCPSolution,
    samples_per_interval: int = 40,
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


def save_outputs(
    case: TestCase,
    OCPSolution: OCPSolution,
    history: list[dict[str, float]],
    defects: dict[str, np.ndarray],
    options: HAdaptiveOptions,
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    t_days = OCPSolution.mesh * case.tof_days
    u_norm_n = np.linalg.norm(OCPSolution.u, axis=0) * case.thrust_unit
    t_dense_days, x_dense, u_dense, sigma_dense = interpolate_OCPSolution(case, OCPSolution)
    u_dense_norm_n = np.linalg.norm(u_dense, axis=0) * case.thrust_unit
    moon_relative_position_nd = x_dense[0:3] - np.array([[1.0 - case.mu], [0.0], [0.0]])
    moon_center_distance_nd = np.linalg.norm(moon_relative_position_nd, axis=0)
    moon_distance_surface_km = (moon_center_distance_nd - case.moon_radius_lu) * case.length_unit
    (
        x_integrated,
        position_error_nd,
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
        position_error_nd=position_error_nd,
        position_error_m=position_error_m,
        velocity_error_nd=velocity_error_nd,
        velocity_error_m_per_s=velocity_error_m_per_s,
        mass_consumption_difference_kg=mass_consumption_difference_kg,
        mass_consumption_abs_difference_kg=mass_consumption_abs_difference_kg,
        interval_defect_scaled=defects["scaled"],
        interval_defect_position_m=defects["position"],
        interval_defect_velocity_m_per_s=defects["velocity"],
        interval_defect_mass_kg=defects["mass_kg"],
        history=np.array(history, dtype=object),
    )

    header = "t_days,x,y,z,vx,vy,vz,m_nd,ux_N,uy_N,uz_N,sigma_N,thrust_norm_N"
    csv_content = np.column_stack(
        [
            t_dense_days,
            x_dense.T,
            u_dense.T * case.thrust_unit,
            sigma_dense.T * case.thrust_unit,
            u_dense_norm_n,
        ]
    )
    csv_path = output_prefix.with_suffix(".csv")
    np.savetxt(csv_path, csv_content, delimiter=",", header=header, comments="")

    lagrange_points = get_collinear_lagrange_points(case)
    departure_orbit = None
    target_orbit = None
    if case.departure_period_nd is not None:
        departure_orbit = propagate_periodic_orbit(case, case.x0_augmented_state, case.departure_period_nd)
    if case.target_period_nd is not None:
        target_orbit = propagate_periodic_orbit(case, case.xf_augmented_state, case.target_period_nd)

    plotter = Plotter(output_prefix)
    plotter.save_projection_figure(case, x_dense, u_dense, departure_orbit, target_orbit, lagrange_points)
    plotter.save_3d_plot(case, x_dense, u_dense, departure_orbit, target_orbit, lagrange_points)
    plotter.save_thrust_plot(t_days, u_norm_n, case.max_thrust_n)
    plotter.save_moon_distance_plot(t_dense_days, moon_distance_surface_km)
    plotter.save_verification_errors(
        t_dense_days,
        position_error_m,
        velocity_error_m_per_s,
        mass_consumption_abs_difference_kg,
    )
    plotter.save_defect_plot(OCPSolution.mesh, case.tof_days, defects["scaled"], options.defect_tolerance)


def h_adaptive_method(
    case: TestCase,
    options: HAdaptiveOptions,
    max_iter: int,
    print_level: int,
    log_prefix: str = "",
) -> tuple[OCPSolution, list[dict[str, float]], dict[str, np.ndarray]]:
    mesh = np.linspace(0.0, 1.0, options.initial_intervals + 1)
    degrees = np.full(options.initial_intervals, options.radau_degree, dtype=int)
    guess: INITIAL_GUESS = None
    history: list[dict[str, float]] = []
    ocp_solution: OCPSolution | None = None
    defects: dict[str, np.ndarray] | None = None

    print(
        f"\n{log_prefix}Hermite-Simpson collocation method warm start: nodes={options.initial_intervals}",
        flush=True,
    )
    hs_solution = _warm_start(
        case=case,
        nodes=options.initial_intervals,
        sol=None,
        max_iter=max_iter,
        print_level=print_level,
    )
    print(
        log_prefix
        + "  objective={fuel_consumed_kg:.6f} kg, "
        "terminal error={terminal_error_nd:.3e}".format(**hs_solution.diagnostics),
        flush=True,
    )
    guess = hs_solution

    for adapt_iter in range(1, options.max_adapt_iterations + 1):
        print(
            f"\n{log_prefix} Iteration {adapt_iter}: "
            f"intervals={degrees.size}, fixed Radau degree={options.radau_degree}",
            flush=True,
        )
        ocp_solution = solve_ocp(
            case=case,
            mesh=mesh,
            degrees=degrees,
            guess=guess,
            max_iter=max_iter,
            print_level=print_level,
            log_prefix=log_prefix,
        )
        defects = estimate_interval_defects(case, ocp_solution, options.rk4_substeps_per_interval)
        max_defect = float(np.max(defects["scaled"]))
        mean_defect = float(np.mean(defects["scaled"]))
        max_defect_idx = int(np.argmax(defects["scaled"]))

        diag = dict(ocp_solution.diagnostics)
        diag["adapt_iteration"] = float(adapt_iter)
        diag["max_interval_defect"] = max_defect
        diag["mean_interval_defect"] = mean_defect
        diag["max_defect_interval"] = float(max_defect_idx)
        history.append(diag)

        print(
            log_prefix
            + "  objective={fuel_consumed_kg:.6f} kg, intervals={intervals:.0f}, "
            " max defect={max_interval_defect:.3e}, "
            "terminal_error={terminal_error_nd:.3e}".format(**diag),
            flush=True,
        )

        if max_defect <= options.defect_tolerance:
            print(f"{log_prefix} tolerance reached.", flush=True)
            break

        new_mesh, new_degrees, refinement = refine_h_mesh(ocp_solution, defects["scaled"], options)
        print(
            f"{log_prefix}  h-refine: split={refinement['split_count']:.0f}",
            flush=True,
        )
        if refinement["changed_count"] <= 0.0:
            print(f"{log_prefix}  h-refinement limit reached.", flush=True)
            break

        guess = ocp_solution
        mesh = new_mesh
        degrees = new_degrees

    return ocp_solution, history, defects

def run_test_case(test_case_id: str) -> str:
    case = CASE_REGISTRY[test_case_id]()
    options = HAdaptiveOptions()
    output_prefix = DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_PREFIX / case.test_case_id
    
    log_prefix = f"[{case.test_case_id}] "

    ocp_solution, history, defects = h_adaptive_method(
        case=case,
        options=options,
        max_iter=MAX_ITER,
        print_level=PRINT_LEVEL,
        log_prefix=log_prefix,
    )
    save_outputs(case, ocp_solution, history, defects, options, output_prefix)

    return case.test_case_id


def main() -> None:
    process_count = min(len(CASE_REGISTRY), mp.cpu_count())
    print(
        f"Running {len(CASE_REGISTRY)} test cases in parallel "
        f"with {process_count} processes.",
        flush=True,
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=process_count, mp_context=context) as executor:
        futures = {executor.submit(run_test_case, test_case_id): test_case_id for test_case_id in CASE_REGISTRY}
        for future in as_completed(futures):
            test_case_id = futures[future]
            try:
                finished_case_id = future.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"Case '{test_case_id}' failed.") from exc
            print(f"\n[{finished_case_id}] Finished.", flush=True)

if __name__ == "__main__":
    main()
