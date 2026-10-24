"""Four-phase deterministic low-thrust transcription with two free SOI exits.

No SPICE imports or queries: all event data and differentiable ephemerides are
verified, stored inputs. States are scaled separately in each central-body
phase. Physical durations multiply every dynamics/mass-flow equation.
"""

from __future__ import annotations
from psyche_mission_events_reconstruction.paths import DATA_ROOT, resolve_input
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
import casadi as ca
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from psyche_mission_events_reconstruction.arrival_boundary import (
    ACTIVE_ARRIVAL_BOUNDARY,
    load_arrival_boundary,
)
from .rk import rk7
from psyche_mission_events_reconstruction.ephemeris_cache import EphemerisCache
from psyche_mission_events_reconstruction.mission_soi import (
    ACTIVE_MARS_BOUNDARY,
    AU_M,
    fixed_mission_soi,
)

ROOT = Path(__file__).resolve().parents[1]
DAY = 86400.0


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class Settings:
    wet_mass_kg: float = 2747.0
    mass_floor_kg: float = 1662.0
    thrust_max_n: float = 0.2793
    isp_s: float = 1794.0
    earth_duration_bounds_days: tuple = (0.05, 10.0)
    mars_duration_bounds_days: tuple = (0.05, 10.0)
    outward_speed_margin_m_s: float = 0.0
    earth_radius_m: float = 6378136.6
    mars_radius_m: float = 3396190.0
    sun_radius_m: float = 695700000.0
    initial_intervals_per_phase: tuple[int, int, int, int] = (60, 50, 90, 50)
    seed_thrust_component_n: float = 0.0001
    seed_substeps_per_interval: int = 2
    ipopt_tolerance: float = 1e-09
    ipopt_constraint_tolerance: float = 1e-10
    ipopt_complementarity_tolerance: float | None = None
    ipopt_mu_min: float | None = None
    ipopt_mu_target: float | None = None
    defect_tolerance: float = 5e-14
    defect_substeps_per_interval: int = 64
    global_position_tolerance_m: float = 10.0
    global_velocity_tolerance_m_s: float = 0.001
    max_intervals: int = 1800
    max_refinements: int = 8
    max_iter: int = 5000
    print_level: int = 5
    hessian_approximation: str = "exact"
    mass_flow_model: str = "throttle_direction"

    def __post_init__(self):
        if self.mass_flow_model != "throttle_direction":
            raise ValueError("Only throttle_direction is supported")
        if (
            not isinstance(self.max_refinements, int)
            or isinstance(self.max_refinements, bool)
            or self.max_refinements < 0
        ):
            raise ValueError("Mesh refinement count must be a nonnegative integer")
        if not (
            0 < self.mass_floor_kg < self.wet_mass_kg and self.thrust_max_n > 0 and (self.isp_s > 0)
        ):
            raise ValueError("Invalid spacecraft properties")
        if any(
            (
                not 0 < a < b
                for a, b in (self.earth_duration_bounds_days, self.mars_duration_bounds_days)
            )
        ):
            raise ValueError("Invalid free-exit duration bounds")
        if len(self.initial_intervals_per_phase) != 4 or any(
            (
                not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2
                for n in self.initial_intervals_per_phase
            )
        ):
            raise ValueError("Specify four integer initial interval counts, each at least two")
        if self.seed_substeps_per_interval < 1 or self.defect_substeps_per_interval < 1:
            raise ValueError("RK7 substep counts must be positive")
        if not 0 <= np.sqrt(3.0) * self.seed_thrust_component_n <= self.thrust_max_n:
            raise ValueError("Initial thrust must fit within engine capacity")
        if any(
            (
                not np.isfinite(t) or t <= 0
                for t in (
                    self.ipopt_tolerance,
                    self.ipopt_constraint_tolerance,
                    self.defect_tolerance,
                )
            )
        ):
            raise ValueError("Numerical tolerances must be positive and finite")
        for value in (self.ipopt_complementarity_tolerance, self.ipopt_mu_min):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(
                    "IPOPT complementarity tolerance and mu_min must be positive and finite"
                )
        if self.ipopt_mu_target is not None and (
            not np.isfinite(self.ipopt_mu_target) or self.ipopt_mu_target < 0
        ):
            raise ValueError("IPOPT mu_target must be nonnegative and finite")


@dataclass(frozen=True)
class Scale:
    length: float
    mu: float
    mass: float

    @property
    def time(self):
        return np.sqrt(self.length**3 / self.mu)

    @property
    def velocity(self):
        return self.length / self.time

    @property
    def state(self):
        return np.r_[np.full(3, self.length), np.full(3, self.velocity), self.mass]


class Mission:

    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.arrival = load_arrival_boundary()
        self.mars = json.loads(ACTIVE_MARS_BOUNDARY.read_text(encoding="utf-8"))
        self.tip_path = DATA_ROOT / "tip/tip_result.json"
        self.tip = json.loads(self.tip_path.read_text(encoding="utf-8"))
        if (
            self.mars["status"] != "verified fixed boundary extracted"
            or self.mars["spacecraft_id"] != -255
            or self.mars["origin_id"] != 499
            or (self.mars["frame"] != "J2000")
            or (self.mars["soi_model"]["convention"] != "fixed_semimajor_axis")
        ):
            raise ValueError("Invalid Mars boundary")
        self.cache_path = resolve_input(self.mars["planetary_cache"]["path"])
        if digest(self.cache_path) != self.mars["planetary_cache"]["sha256"]:
            raise ValueError("Changed planetary ephemerides")
        self.cache = EphemerisCache.load(self.cache_path)
        if self.cache.metadata["tip_result_sha256"] != digest(self.tip_path):
            raise ValueError("TIP/cache mismatch")
        self.t0 = self.tip["epochs"]["tip"]["et_tdb_s"]
        self.tm = self.mars["epoch"]["et_tdb_s"]
        self.tf = self.arrival["epoch"]["et_tdb_s"]
        self.x0 = np.r_[self.tip["earth_relative_state_si"], self.settings.wet_mass_kg]
        self.xm = np.asarray(self.mars["mars_relative_state_si"])
        self.xf = np.asarray(self.arrival["sun_relative_state_si"])
        if (
            not self.cache.valid_interval[0]
            <= self.t0
            < self.tm
            < self.tf
            <= self.cache.valid_interval[1]
        ):
            raise ValueError("Event epochs outside cached ephemerides")
        self.origins = (399, 10, 499, 10)
        mu = self.cache.metadata["mu_m3_s2"]
        self.scales = [
            Scale(AU_M if b == 10 else 100000000.0, mu[str(b)], self.settings.wet_mass_kg)
            for b in self.origins
        ]
        self.soi = {b: fixed_mission_soi(b, mu).radius_rate(None)[0] for b in (399, 499)}
        self.body_functions = {b: self.cache.casadi_functions(b)[0] for b in (399, 499)}
        self.seed_path = DATA_ROOT / "rollout_timing/four_phase_seed.npz"
        seed_result_path = self.seed_path.with_name("result.json")
        self.seed_result = json.loads(seed_result_path.read_text(encoding="utf-8"))
        provenance = self.seed_result["provenance"]
        for key, path in (
            ("tip_sha256", self.tip_path),
            ("mars_entry_sha256", ACTIVE_MARS_BOUNDARY),
            ("arrival_sha256", ACTIVE_ARRIVAL_BOUNDARY),
        ):
            if provenance[key] != digest(path):
                raise ValueError("Rollout seed has stale boundary provenance: " + key)
        with np.load(self.seed_path, allow_pickle=False) as data:
            self.seed_exit_days = np.array(
                [
                    (data["phase1_epochs_et_tdb_s"][-1] - self.t0) / DAY,
                    (data["phase3_epochs_et_tdb_s"][-1] - self.tm) / DAY,
                ]
            )
        self.seed_thrust_n = np.full(3, self.settings.seed_thrust_component_n)
        self.seed_sigma_n = self.mass_flow_thrust(self.seed_thrust_n)
        self.seed = self.make_initial_seed()
        self.seed_splines = []
        for i, scale in enumerate(self.scales, 1):
            times = self.seed[f"phase{i}_elapsed_s"]
            states = self.seed[f"phase{i}_states_si"]
            derivatives = np.array(
                [self.rhs_si(i - 1, y, self.seed_thrust_n, self.seed_sigma_n) for y in states]
            )
            self.seed_splines.append(
                CubicHermiteSpline(times / times[-1], states, derivatives * times[-1])
            )

    def make_initial_seed(self):
        """Two RK7 steps per initial arc, with 1e-4 N on each J2000 axis.

        The Mars chain is anchored at its fixed Cartesian entry state; mass
        comes from the Earth chain. Downstream target mismatches are retained.
        Guessed exit epochs are not forced to be exact crossings of this seed.
        """
        epochs, arrays = (self.epochs(self.seed_exit_days), {})
        state = self.x0.copy()
        for phase, mesh in enumerate(self.initial_meshes()):
            if phase == 2:
                state[:6] = self.xm
            duration = epochs[phase + 1] - epochs[phase]
            history = [state.copy()]
            rhs = lambda y: self.rhs_si(phase, y, self.seed_thrust_n, self.seed_sigma_n)
            for fraction in np.diff(mesh):
                dt = duration * fraction / self.settings.seed_substeps_per_interval
                for _ in range(self.settings.seed_substeps_per_interval):
                    state = rk7(rhs, state, dt)
                if not np.all(np.isfinite(state)):
                    raise RuntimeError(
                        "Nonfinite RK7 initial rollout; no seed substitution was made"
                    )
                history.append(state.copy())
            prefix = f"phase{phase + 1}_"
            arrays.update(
                {
                    prefix + "tau": mesh,
                    prefix + "elapsed_s": duration * mesh,
                    prefix + "epochs_et_tdb_s": epochs[phase] + duration * mesh,
                    prefix + "states_si": np.array(history),
                    prefix + "thrust_n": np.tile(self.seed_thrust_n, (len(mesh) - 1, 1)),
                    prefix + "sigma_n": np.full(len(mesh) - 1, self.seed_sigma_n),
                }
            )
            if phase in (0, 2):
                state[:6] += self.cache.state(399 if phase == 0 else 499, epochs[phase + 1])
        return arrays

    def epochs(self, exit_days):
        return (
            self.t0,
            self.t0 + DAY * exit_days[0],
            self.tm,
            self.tm + DAY * exit_days[1],
            self.tf,
        )

    def mass_flow_thrust(self, thrust):
        return float(np.linalg.norm(thrust))

    def rhs(self, phase, x, u, sigma):
        """Nondimensional state derivative in the phase's Kepler time unit.

        u and sigma are fractions of engine capacity. The tiny squared-radius
        regularizer only avoids evaluation at an exact zero trial vector.
        """
        s, cfg = (self.scales[phase], self.settings)
        acceleration_factor = cfg.thrust_max_n * s.time**2 / (s.mass * s.length)
        mass_factor = cfg.thrust_max_n * s.time / (s.mass * cfg.isp_s * 9.80665)
        radius = ca.sqrt(ca.dot(x[:3], x[:3]) + 1e-30)
        return ca.vertcat(
            x[3:6], -x[:3] / radius**3 + acceleration_factor * u / x[6], -mass_factor * sigma
        )

    def rhs_si(self, phase, x, thrust, sigma):
        radius = np.linalg.norm(x[:3])
        if radius <= 0 or x[6] <= 0:
            raise ValueError("Singular propagated state")
        return np.r_[
            x[3:6],
            -self.scales[phase].mu * x[:3] / radius**3 + thrust / x[6],
            -sigma / (9.80665 * self.settings.isp_s),
        ]

    def initial_meshes(self):
        return [np.linspace(0.0, 1.0, n + 1) for n in self.settings.initial_intervals_per_phase]

    def provenance(self):
        return {
            "tip_sha256": digest(self.tip_path),
            "mars_sha256": digest(ACTIVE_MARS_BOUNDARY),
            "arrival_sha256": digest(ACTIVE_ARRIVAL_BOUNDARY),
            "cache_sha256": digest(self.cache_path),
            "rollout_seed_sha256": digest(self.seed_path),
            "archived_seed_usage": "Earth/Mars coast exit epochs only, as free-time initial guesses",
            "initial_guess_policy": "small_thrust_rk7_two_substeps_v1",
            "mass_flow_model": self.settings.mass_flow_model,
            "geometric_path_policy": "central_body_surface_exclusion_only; no SOI path inequalities",
            "exit_direction_policy": "zero radial-speed threshold; strict positivity reported separately",
            "ephemerides": "stored differentiable cubic splines; no SPICE calls in optimiser",
            "spacecraft_parameters_source": "Thesis SYS-REQ-06/07/10; 1662 kg floor assumes all 1085 kg xenon is available",
            "radii_source": "https://ssd.jpl.nasa.gov/planets/phys_par.html",
            "numerical_method_source": "HS warm start; uniform RK7 multiple shooting; local-defect h-refinement; throttle and unit direction",
            "added_assumptions": "Planet duration bounds 0.05-10 d each and central-body spherical surface exclusion. No positive outward-speed margin and no inside/outside SOI path constraints.",
        }


def evaluate_phase(solution, phase, tau, mission):
    if solution["method"] == "shooting":
        from .flows import evaluate_shooting_phase

        return evaluate_shooting_phase(solution, phase, tau, mission)
    record = solution["phases"][phase]
    mesh, states = (record["mesh"], record["states_si"])
    k = min(max(np.searchsorted(mesh, tau, side="right") - 1, 0), len(mesh) - 2)
    theta = (tau - mesh[k]) / (mesh[k + 1] - mesh[k])
    a, b = (states[k], states[k + 1])
    h = record["duration_s"] * (mesh[k + 1] - mesh[k])
    u, sigma = (record["thrust_n"][k], record["sigma_n"][k])
    fa, fb = (mission.rhs_si(phase, a, u, sigma), mission.rhs_si(phase, b, u, sigma))
    return (
        (2 * theta**3 - 3 * theta**2 + 1) * a
        + (theta**3 - 2 * theta**2 + theta) * h * fa
        + (-2 * theta**3 + 3 * theta**2) * b
        + (theta**3 - theta**2) * h * fb
    )


def throttle_direction_seed(thrust_n, thrust_max_n):
    """Preserve every nonzero thrust; zero thrust has an arbitrary free direction.

    No near-zero threshold, clipping, coast fixing or thrust floor is imposed.
    Returned directions are rows, matching the archived SI thrust arrays.
    """
    thrust = np.asarray(thrust_n, dtype=float)
    magnitudes = np.linalg.norm(thrust, axis=1)
    directions = np.zeros_like(thrust)
    directions[:, 0] = 1.0
    np.divide(thrust, magnitudes[:, None], out=directions, where=magnitudes[:, None] > 0.0)
    return (magnitudes / thrust_max_n, directions)


class Transcription:

    def __init__(
        self,
        mission,
        meshes,
        method="shooting",
        guess=None,
        objective="fuel",
        barrier="adaptive",
        shooting_substeps=20,
    ):
        if method not in ("hs", "shooting") or objective not in ("fuel", "energy"):
            raise ValueError("Unknown method or warm-start objective")
        self.mission, self.meshes, self.method = (mission, meshes, method)
        self.objective, self.barrier = (objective, barrier)
        self.shooting_substeps = shooting_substeps
        cfg = mission.settings
        self.opti = op = ca.Opti()
        self.exit_days = op.variable(2)
        for i, (lo, hi) in enumerate(
            (cfg.earth_duration_bounds_days, cfg.mars_duration_bounds_days)
        ):
            op.subject_to(op.bounded(lo, self.exit_days[i], hi))
        self.epochs = mission.epochs(self.exit_days)
        self.variables = []
        self.direction_variables = []
        self.group_ranges = {}
        self._constraint_count = int(op.ng)
        self.energy = 0.0

        def add(name, expression):
            start = self._constraint_count
            op.subject_to(expression)
            self._constraint_count += int(expression.numel())
            self.group_ranges.setdefault(name, []).append((start, self._constraint_count))

        def path(phase, x):
            scale, origin = (mission.scales[phase], mission.origins[phase])
            add("mass_bounds", x[6] >= cfg.mass_floor_kg / cfg.wet_mass_kg)
            min_radius = {399: cfg.earth_radius_m, 499: cfg.mars_radius_m, 10: cfg.sun_radius_m}[
                origin
            ]
            add("surface_exclusion", ca.sumsqr(x[:3]) >= (min_radius / scale.length) ** 2)

        for phase, (mesh, scale) in enumerate(zip(meshes, mission.scales)):
            if method == "shooting":
                from .flows import get_arc_flow

                flow = get_arc_flow(mission, phase, shooting_substeps)
            if mesh[0] != 0 or mesh[-1] != 1 or np.any(np.diff(mesh) <= 0):
                raise ValueError("Mesh must increase from zero to one")
            n = len(mesh) - 1
            X = op.variable(7, n + 1)
            D, S = (op.variable(3, n), op.variable(1, n))
            U = D * ca.repmat(S, 3, 1)
            self.direction_variables.append(D)
            stages = []
            start, end = (self.epochs[phase], self.epochs[phase + 1])
            duration = end - start
            add("positive_duration", duration >= 1.0)
            for k in range(n + 1):
                path(phase, X[:, k])
            for k, fraction in enumerate(np.diff(mesh)):
                add("thrust_capacity", op.bounded(0.0, S[0, k], 1.0))
                add("unit_direction", ca.sumsqr(D[:, k]) == 1.0)
                h = duration * fraction / scale.time
                squared_effort = ca.sumsqr(U[:, k])
                self.energy += duration * fraction / DAY * 0.5 * squared_effort
                if method == "hs":
                    fa = mission.rhs(phase, X[:, k], U[:, k], S[0, k])
                    fb = mission.rhs(phase, X[:, k + 1], U[:, k], S[0, k])
                    middle = 0.5 * (X[:, k] + X[:, k + 1]) + h / 8 * (fa - fb)
                    fm = mission.rhs(phase, middle, U[:, k], S[0, k])
                    path(phase, middle)
                    add("dynamics", X[:, k + 1] - X[:, k] == h / 6 * (fa + 4 * fm + fb))
                elif method == "shooting":
                    propagated, samples = flow(X[:, k], U[:, k], S[0, k], h)
                    for j in range(samples.shape[1]):
                        path(phase, samples[:, j])
                    add("dynamics", X[:, k + 1] == propagated)
            self.variables.append((X, U, S, stages))
        X = [v[0] for v in self.variables]
        add("fixed_tip", X[0][:, 0] == mission.x0 / mission.scales[0].state)
        add("fixed_mars_entry", X[2][:6, 0] == mission.xm / mission.scales[2].state[:6])
        add("fixed_arrival", X[3][:6, -1] == mission.xf / mission.scales[3].state[:6])
        for left, right, body, event in ((0, 1, 399, 1), (1, 2, 499, 2), (2, 3, 499, 3)):
            a = X[left][:6, -1] * ca.DM(mission.scales[left].state[:6])
            b = X[right][:6, 0] * ca.DM(mission.scales[right].state[:6])
            planet = mission.body_functions[body](self.epochs[event])
            difference = a + planet - b if left in (0, 2) else a - planet - b
            add("frame_links", difference / ca.DM(mission.scales[1].state[:6]) == 0)
            add("mass_links", X[left][6, -1] == X[right][6, 0])
        for phase, body in ((0, 399), (2, 499)):
            scale = mission.scales[phase]
            state = X[phase][:, -1]
            add("exit_surface", ca.sumsqr(state[:3]) == (mission.soi[body] / scale.length) ** 2)
            add(
                "exit_direction",
                ca.dot(state[:3], state[3:6])
                >= mission.soi[body] / scale.length * cfg.outward_speed_margin_m_s / scale.velocity,
            )
        self.fuel_fraction = 1.0 - X[-1][6, -1]
        if self._constraint_count != int(op.ng):
            raise AssertionError("Constraint diagnostics row accounting mismatch")
        objectives = {
            "fuel": self.fuel_fraction,
            "energy": self.energy / ((mission.tf - mission.t0) / DAY),
            "feasibility": 0.0,
        }
        op.minimize(objectives[objective])
        op.set_initial(
            self.exit_days, mission.seed_exit_days if guess is None else guess["exit_days"]
        )
        for phase, (mesh, scale, (x, u, s, stages)) in enumerate(
            zip(meshes, mission.scales, self.variables)
        ):
            sample = (
                (lambda tau: mission.seed_splines[phase](tau))
                if guess is None
                else lambda tau, p=phase: evaluate_phase(guess, p, tau, mission)
            )
            op.set_initial(x, np.array([sample(t) / scale.state for t in mesh]).T)
            if guess is None:
                initial_thrust = np.tile(mission.seed_thrust_n, (len(mesh) - 1, 1))
            else:
                old = guess["phases"][phase]
                ids = np.clip(
                    np.searchsorted(old["mesh"], 0.5 * (mesh[:-1] + mesh[1:]), side="right") - 1,
                    0,
                    len(old["mesh"]) - 2,
                )
                initial_thrust = old["thrust_n"][ids]
            if (
                guess is not None
                and guess.get("mass_flow_model") == "throttle_direction"
                and ("throttle" in old)
                and ("direction" in old)
            ):
                throttle, direction = (old["throttle"][ids], old["direction"][ids])
            else:
                throttle, direction = throttle_direction_seed(initial_thrust, cfg.thrust_max_n)
            op.set_initial(s, throttle[None])
            op.set_initial(self.direction_variables[phase], direction.T)
        self.ipopt_options = {
            "max_iter": cfg.max_iter,
            "tol": cfg.ipopt_tolerance,
            "constr_viol_tol": cfg.ipopt_constraint_tolerance,
            "acceptable_iter": 0,
            "bound_relax_factor": 0.0,
            "mu_strategy": barrier,
            "print_level": cfg.print_level,
            "sb": "yes",
            "hessian_approximation": cfg.hessian_approximation,
            "nlp_scaling_method": "gradient-based",
            "bound_push": 1e-08,
            "bound_frac": 1e-08,
        }
        for name, value in (
            ("compl_inf_tol", cfg.ipopt_complementarity_tolerance),
            ("mu_min", cfg.ipopt_mu_min),
            ("mu_target", cfg.ipopt_mu_target),
        ):
            if value is not None:
                self.ipopt_options[name] = value
        op.solver(
            "ipopt", {"expand": method != "shooting", "print_time": False}, self.ipopt_options
        )

    def solve(self, checkpoint_directory=None):
        if checkpoint_directory is not None:
            from .checkpoint import IterationCheckpoint

            checkpoint = IterationCheckpoint(self, checkpoint_directory)
            self.opti.callback(checkpoint)
        start = time.perf_counter()
        try:
            sol = self.opti.solve()
            success, error = (True, None)
        except RuntimeError as exc:
            sol, success, error = (self.opti.debug, False, str(exc))
        stats = self.opti.stats()
        value = lambda x: np.asarray(sol.value(x), dtype=float)
        exits = value(self.exit_days).ravel()
        epochs = self.mission.epochs(exits)
        phases = []
        for i, (mesh, (x, u, s, stages), scale) in enumerate(
            zip(self.meshes, self.variables, self.mission.scales)
        ):
            phases.append(
                {
                    "mesh": np.asarray(mesh),
                    "states_si": value(x).T * scale.state,
                    "thrust_n": value(u).reshape(3, -1).T * self.mission.settings.thrust_max_n,
                    "sigma_n": value(s).ravel() * self.mission.settings.thrust_max_n,
                    "stages_si": np.array(
                        [value(stage).reshape(7, -1).T * scale.state for stage in stages]
                    ),
                    "start_et": float(epochs[i]),
                    "duration_s": float(epochs[i + 1] - epochs[i]),
                }
            )
            if self.direction_variables[i] is not None:
                phases[-1].update(
                    throttle=value(s).ravel(),
                    direction=value(self.direction_variables[i]).reshape(3, -1).T,
                )
        constraints, lo, hi = (
            value(self.opti.g).ravel(),
            value(self.opti.lbg).ravel(),
            value(self.opti.ubg).ravel(),
        )
        violation = np.maximum(np.maximum(lo - constraints, constraints - hi), 0.0)
        groups = {
            key: float(max((np.max(violation[a:b], initial=0) for a, b in ranges)))
            for key, ranges in self.group_ranges.items()
        }
        energy_n2_s = sum(
            (
                0.5
                * p["duration_s"]
                * np.dot(np.diff(p["mesh"]), np.sum(p["thrust_n"] ** 2, axis=1))
                for p in phases
            )
        )
        fuel_kg = float(value(self.fuel_fraction)) * self.mission.settings.wet_mass_kg
        return {
            "method": self.method,
            "objective": self.objective,
            "barrier": self.barrier,
            "ipopt_options": self.ipopt_options.copy(),
            "mass_flow_model": self.mission.settings.mass_flow_model,
            "sigma_semantics": "Tmax times throttle alpha; thrust=Tmax*alpha*unit_direction; no epigraph slack",
            "shooting_substeps": self.shooting_substeps if self.method == "shooting" else None,
            "shooting_grid": "uniform" if self.method == "shooting" else None,
            "solver_success": success,
            "solver_status": stats["return_status"],
            "solver_error": error,
            "iterations": stats.get("iter_count"),
            "wall_seconds": time.perf_counter() - start,
            "last_ipopt_metrics": {
                key: float(values[-1])
                for key, values in stats.get("iterations", {}).items()
                if isinstance(values, (list, tuple)) and values
            },
            "exit_days": exits,
            "phases": phases,
            "fuel_consumed_kg": fuel_kg,
            "energy_surrogate_n2_s": float(energy_n2_s),
            "energy_cost_n2_s": float(energy_n2_s),
            "energy_definition": "0.5 integral dot(T,T) dt",
            "objective_value": {"fuel": fuel_kg, "energy": float(energy_n2_s), "feasibility": 0.0}[
                self.objective
            ],
            "objective_units": {"fuel": "kg", "energy": "N^2 s", "feasibility": "dimensionless"}[
                self.objective
            ],
            "max_scaled_constraint_violation": float(np.max(violation)),
            "constraint_groups": groups,
            "variables": self.opti.nx,
            "constraints": self.opti.ng,
        }


def save_solution(solution, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"exit_days": solution["exit_days"]}
    metadata = {k: v for k, v in solution.items() if k not in ("phases", "exit_days")}
    metadata["phases"] = []
    for i, phase in enumerate(solution["phases"]):
        metadata["phases"].append({k: v for k, v in phase.items() if not isinstance(v, np.ndarray)})
        arrays.update({f"p{i}_{k}": v for k, v in phase.items() if isinstance(v, np.ndarray)})
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_solution(path):
    with np.load(path, allow_pickle=False) as data:
        result = json.loads(str(data["metadata"]))
        result["exit_days"] = data["exit_days"]
        for i, phase in enumerate(result["phases"]):
            phase.update(
                {k.removeprefix(f"p{i}_"): data[k] for k in data.files if k.startswith(f"p{i}_")}
            )
    return result
