"""Single-phase CR3BP and Earth-Mars HS/shooting solves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

from . import models as reference
import integrator
from .io import digest
from .psyche import throttle_direction_seed
from .io import write_json
from .flows import arc_flow
from .checkpoint import IterationCheckpoint


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_intervals(case, objective):
    return 50


def validate_mesh(mesh):
    mesh = np.asarray(mesh, dtype=float)
    if (
        mesh.ndim != 1
        or len(mesh) < 3
        or not np.all(np.isfinite(mesh))
        or mesh[0] != 0.0
        or mesh[-1] != 1.0
        or np.any(np.diff(mesh) <= 0.0)
    ):
        raise ValueError("Mesh must increase strictly from zero to one with at least two arcs")
    return mesh.copy()


def refine_mesh(mesh, relative, tolerance, max_intervals):
    """Bisect failing intervals, largest defect first if capacity is limited."""
    mesh = validate_mesh(mesh)
    relative = np.asarray(relative)
    if relative.shape != (len(mesh) - 1,) or not np.all(np.isfinite(relative)):
        raise ValueError("One finite local defect is required per arc")
    capacity = max(0, max_intervals - (len(mesh) - 1))
    selected = sorted(np.flatnonzero(relative > tolerance), key=lambda k: (-relative[k], k))[
        :capacity
    ]
    middle = [(mesh[k] + mesh[k + 1]) / 2.0 for k in selected]
    return validate_mesh(np.sort(np.r_[mesh, middle])), selected


def resample_candidate(case, arrays, mesh):
    """Preserve old nodes and controls; use cubic Hermite states at new nodes."""
    old = validate_mesh(arrays["mesh"])
    mesh = validate_mesh(mesh)
    if not np.all(np.isin(old, mesh)):
        raise ValueError("Refinement must retain every old mesh boundary")
    x = np.empty((7, len(mesh)))
    for j, tau in enumerate(mesh):
        node = np.searchsorted(old, tau)
        if node < len(old) and old[node] == tau:
            x[:, j] = arrays["states_nd"][:, node]
            continue
        k = node - 1
        theta = (tau - old[k]) / (old[k + 1] - old[k])
        h = case.tof_nd * (old[k + 1] - old[k])
        xa, xb = arrays["states_nd"][:, k : k + 2].T
        u = case.max_thrust_nd * arrays["throttle"][k] * arrays["direction"][:, k]
        sigma = case.max_thrust_nd * arrays["throttle"][k]
        fa, fb = reference.eom(case, xa, u, sigma), reference.eom(case, xb, u, sigma)
        x[:, j] = (
            (2 * theta**3 - 3 * theta**2 + 1) * xa
            + (theta**3 - 2 * theta**2 + theta) * h * fa
            + (-2 * theta**3 + 3 * theta**2) * xb
            + (theta**3 - theta**2) * h * fb
        )
    ids = np.searchsorted(old, 0.5 * (mesh[:-1] + mesh[1:]), side="right") - 1
    return x, arrays["direction"][:, ids].copy(), arrays["throttle"][ids].copy()


def is_cr3bp(case):
    return isinstance(case, reference.CR3BPEarthMoon)


def default_shooting_substeps(case):
    return 2 if is_cr3bp(case) else 20


def default_defect_tolerance(case):
    return 5e-13 if is_cr3bp(case) else 1e-13


def moon_clearance_km(case, states):
    """Geometric clearance in the native barycentric rotating coordinates."""
    centre = np.array([1.0 - case.mu, 0.0, 0.0])
    return (
        np.linalg.norm(np.asarray(states)[:3] - centre[:, None], axis=0) * case.length_unit
        - case.moon_radius
    )


class MissionAdapter:
    def __init__(self, case):
        self.case = case
        self.settings = SimpleNamespace(
            mass_flow_model="throttle_direction",
            wet_mass_kg=case.m0_wet,
            mass_floor_kg=case.m0_dry,
            thrust_max_n=case.max_thrust_n,
            isp_s=case.isp,
        )
        state_scale = np.r_[
            np.full(3, case.length_unit * 1000.0),
            np.full(3, case.velocity_unit * 1000.0),
            case.m0_wet,
        ]
        self.scales = [SimpleNamespace(state=state_scale)]

    def rhs(self, phase, state, normalized_thrust, throttle):
        if phase != 0:
            raise ValueError("The deterministic cases have a single dynamics phase")
        return reference.eom(
            self.case,
            state,
            self.case.max_thrust_nd * normalized_thrust,
            self.case.max_thrust_nd * throttle,
        )


class ShootingProblem:
    def __init__(
        self,
        case,
        objective,
        ipopt_options,
        intervals=None,
        substeps=None,
        method="shooting",
        incoming=None,
        mesh=None,
    ):
        if objective not in ("fuel", "energy"):
            raise ValueError("Select fuel or energy")
        if method not in ("hs", "shooting"):
            raise ValueError("Select hs or shooting")
        self.case, self.objective, self.method = case, objective, method
        self.substeps = default_shooting_substeps(case) if substeps is None else substeps
        self.moon_surface_nodes = is_cr3bp(case)
        self.mission = MissionAdapter(case)
        if mesh is not None and intervals is not None:
            raise ValueError("Provide either a mesh or an interval count")
        n = default_intervals(case, objective) if intervals is None else intervals
        self.mesh = validate_mesh(np.linspace(0.0, 1.0, n + 1) if mesh is None else mesh)
        n = len(self.mesh) - 1
        self.meshes = [self.mesh]
        self.flow = arc_flow(self.mission, 0, self.substeps)
        self.opti = op = ca.Opti()
        self.X, self.D, self.A = op.variable(7, n + 1), op.variable(3, n), op.variable(1, n)
        self.U = self.D * ca.repmat(self.A, 3, 1)
        self.variables = [(self.X, self.U, self.A, [])]
        self.direction_variables = [self.D]
        self.exit_days = ca.DM.zeros(0, 1)  # Fixed-duration, single-phase cases.
        self.group_ranges, self._rows = {}, 0

        def add(group, expression):
            start = self._rows
            op.subject_to(expression)
            self._rows += expression.numel()
            self.group_ranges.setdefault(group, []).append((start, self._rows))

        # Requested Moon exclusion applies only at discretization boundaries,
        # in both HS and shooting. No interior Moon inequalities are added.
        for k in range(n + 1):
            add("mass_floor", self.X[6, k] >= case.m0_dry_nd)
            if self.moon_surface_nodes:
                moon_offset = self.X[:3, k] - ca.DM([1.0 - case.mu, 0.0, 0.0])
                add("moon_surface_nodes", ca.sumsqr(moon_offset / case.moon_radius_lu) >= 1.0)
        energy = 0.0
        for k, fraction in enumerate(np.diff(self.mesh)):
            add("throttle", op.bounded(0.0, self.A[0, k], 1.0))
            add("unit_direction", ca.sumsqr(self.D[:, k]) == 1.0)
            h = case.tof_nd * fraction
            if method == "hs":
                fa = self.mission.rhs(0, self.X[:, k], self.U[:, k], self.A[0, k])
                fb = self.mission.rhs(0, self.X[:, k + 1], self.U[:, k], self.A[0, k])
                middle = 0.5 * (self.X[:, k] + self.X[:, k + 1]) + h / 8.0 * (fa - fb)
                fm = self.mission.rhs(0, middle, self.U[:, k], self.A[0, k])
                add("mass_floor", middle[6] >= case.m0_dry_nd)
                add("dynamics", self.X[:, k + 1] - self.X[:, k] == h / 6.0 * (fa + 4.0 * fm + fb))
            else:
                end, internal = self.flow(self.X[:, k], self.U[:, k], self.A[0, k], h)
                for j in range(internal.shape[1]):
                    add("mass_floor", internal[6, j] >= case.m0_dry_nd)
                add("dynamics", self.X[:, k + 1] == end)
            energy += 0.5 * fraction * ca.sumsqr(self.U[:, k])
        add("initial_state", self.X[:, 0] == case.x0_augmented_state)
        add("terminal_state", self.X[:6, -1] == case.xf_state)
        self.fuel_fraction, self.normalized_energy = 1.0 - self.X[6, -1], energy
        op.minimize(self.fuel_fraction if objective == "fuel" else energy)
        if int(op.ng) != self._rows:
            raise AssertionError("Constraint row accounting failed")
        self.initial_source = "fresh original dynamics rollout"
        if incoming is None:
            x, u, sigma = reference.rollout_initial_guess(case, self.mesh)
            alpha, direction = throttle_direction_seed(u.T * case.thrust_unit, case.max_thrust_n)
        else:
            result, arrays = incoming
            if (
                method != "shooting"
                or not result["converged"]
                or result["method"] not in ("hs", "shooting")
                or result["objective"] != objective
                or result["case"] != case.test_case_id
            ):
                raise ValueError(
                    "Handoff requires converged same-case, same-objective HS or shooting"
                )
            if result["method"] == "hs":
                np.testing.assert_array_equal(arrays["mesh"], self.mesh)
            x, d, alpha = resample_candidate(case, arrays, self.mesh)
            direction = d.T
            u, sigma = case.max_thrust_nd * d * alpha[None], case.max_thrust_nd * alpha[None]
            self.initial_source = (
                "converged same-objective Hermite-Simpson solution"
                if result["method"] == "hs"
                else "preceding converged shooting mesh; Hermite states and inherited arc controls"
            )
        if not all(np.all(np.isfinite(v)) for v in (x, u, sigma, alpha, direction)):
            raise ValueError("Initial guess contains nonfinite values")
        self.seed = dict(
            mesh=self.mesh,
            states_nd=x,
            control_nd=u,
            magnitude_nd=sigma,
            throttle=alpha,
            direction=direction,
        )
        op.set_initial(self.X, x)
        op.set_initial(self.D, direction.T)
        op.set_initial(self.A, alpha[None])
        self.ipopt_options = dict(ipopt_options)
        op.solver("ipopt", {"expand": False, "print_time": False}, self.ipopt_options)

    def solve(self, directory):
        self.opti.callback(IterationCheckpoint(self, directory / "checkpoints"))
        started = time.perf_counter()
        error = None
        try:
            answer = self.opti.solve()
            success = True
        except RuntimeError as exc:
            answer, success, error = self.opti.debug, False, str(exc)
        stats = self.opti.stats()
        value = lambda expr: np.asarray(answer.value(expr), dtype=float)
        x, d, a = value(self.X), value(self.D).reshape(3, -1), value(self.A).ravel()
        thrust = value(self.U).reshape(3, -1) * self.case.max_thrust_n
        g, lo, hi = (value(expr).ravel() for expr in (self.opti.g, self.opti.lbg, self.opti.ubg))
        violation = np.maximum(np.maximum(lo - g, g - hi), 0.0)
        fuel = (1.0 - x[6, -1]) * self.case.m0_wet
        energy = (
            0.5
            * self.case.tof_days
            * 86400.0
            * np.dot(np.diff(self.mesh), np.sum(thrust**2, axis=0))
        )
        arrays = dict(
            mesh=self.mesh,
            states_nd=x,
            states_si=x * self.mission.scales[0].state[:, None],
            thrust_n=thrust,
            throttle=a,
            direction=d,
            mass_flow_equivalent_n=self.case.max_thrust_n * a,
            primal=value(self.opti.x).ravel(),
            constraint_multipliers=value(self.opti.lam_g).ravel(),
        )
        result = dict(
            case=self.case.test_case_id,
            objective=self.objective,
            method=self.method,
            initialization=self.initial_source,
            converged=success,
            solver_status=stats["return_status"],
            solver_error=error,
            iterations=stats.get("iter_count"),
            solver_seconds=time.perf_counter() - started,
            fuel_kg=float(fuel),
            energy_n2_s=float(energy),
            objective_value=float(fuel if self.objective == "fuel" else energy),
            objective_units="kg" if self.objective == "fuel" else "N^2 s",
            arcs=len(self.mesh) - 1,
            shooting_substeps=self.substeps,
            variables=int(self.opti.nx),
            constraints=int(self.opti.ng),
            max_constraint_violation=float(violation.max()),
            constraint_groups={
                key: float(max(np.max(violation[start:end]) for start, end in spans))
                for key, spans in self.group_ranges.items()
            },
            max_unit_direction_residual=float(np.max(abs(np.sum(d * d, axis=0) - 1.0))),
            max_mass_flow_thrust_difference_n=float(
                np.max(abs(self.case.max_thrust_n * a - np.linalg.norm(thrust, axis=0)))
            ),
            last_ipopt_metrics={
                key: float(values[-1])
                for key, values in stats.get("iterations", {}).items()
                if isinstance(values, (list, tuple)) and values
            },
        )
        if self.moon_surface_nodes:
            clearance = moon_clearance_km(self.case, x) * 1000.0
            result["moon_surface_nodes"] = dict(
                count=x.shape[1],
                radius_km=self.case.moon_radius,
                minimum_clearance_m=float(clearance.min()),
                minimum_clearance_node=int(clearance.argmin()),
                enforced_at="mesh boundaries only, including initial and final nodes",
            )
        np.savez_compressed(directory / "solution.npz", **arrays)
        write_json(directory / "result.json", result)
        return result, arrays


def flow_checks(problem):
    """Independent RK7, DOP853 and finite-difference checks including coast."""
    case = problem.case
    state = case.x0_augmented_state.copy()
    duration = min(case.tof_nd / (len(problem.mesh) - 1), 0.01)
    flow, weights = problem.flow, ca.DM([0.2, -0.3, 0.1, 0.4, -0.2, 0.5, 0.7])
    z = ca.MX.sym("z", 12)  # state, unit direction, throttle, duration
    end, _ = flow(z[:7], z[7:10] * z[10], z[10], z[11])
    f = ca.Function("flow_values", [z], [end])
    jac = ca.Function("flow_jac", [z], [ca.jacobian(end, z)])
    hess = ca.Function("flow_hess", [z], [ca.hessian(ca.dot(weights, end), z)[0]])
    checks = []
    for throttle in (0.0, 1e-8, 0.4):
        point = np.r_[state, [0.6, 0.0, 0.8], throttle, duration]
        analytic = np.asarray(jac(point))
        numerical = np.empty_like(analytic)
        for j in range(12):
            delta = np.zeros(12)
            delta[j] = 1e-6 * max(abs(point[j]), 1.0)
            numerical[:, j] = np.asarray(
                (f(point + delta) - f(point - delta)) / (2 * delta[j])
            ).ravel()
        direction, eps = np.linspace(-0.6, 0.7, 12), 2e-6
        estimated_hv = (
            np.asarray((jac(point + eps * direction) - jac(point - eps * direction)) / (2 * eps)).T
            @ np.asarray(weights).ravel()
        )
        exact_hv = np.asarray(hess(point)) @ direction
        computed = np.asarray(f(point)).ravel()
        u, sigma = (
            case.max_thrust_nd * throttle * np.array([0.6, 0.0, 0.8]),
            case.max_thrust_nd * throttle,
        )
        manual = state.copy()
        for _ in range(problem.substeps):
            manual = integrator.rk7(case, manual, u, sigma, duration / problem.substeps)
        dop = solve_ivp(
            lambda t, y: reference.eom(case, y, u, sigma),
            (0.0, duration),
            state,
            method="DOP853",
            rtol=1e-13,
            atol=1e-14,
            max_step=duration / 32.0,
        )
        if not dop.success:
            raise RuntimeError(dop.message)
        checks.append(
            dict(
                throttle=throttle,
                jacobian_error=float(np.max(abs(analytic - numerical) / (1.0 + abs(analytic)))),
                hessian_vector_error=float(
                    np.max(abs(exact_hv - estimated_hv) / (1.0 + abs(exact_hv)))
                ),
                manual_rk7_difference_nd=float(np.max(abs(computed - manual))),
                dop853_difference_nd=float(np.max(abs(computed - dop.y[:, -1]))),
            )
        )
    return checks


def local_defects(problem, arrays, substeps=64):
    """Original driver's relative r/v/m criterion, evaluated with uniform RK7."""
    case, mesh = problem.case, arrays["mesh"]
    x, thrust, alpha = arrays["states_nd"], arrays["thrust_n"], arrays["throttle"]
    local_flow = arc_flow(problem.mission, 0, substeps)
    defects = np.zeros_like(x[:, 1:])
    for k, fraction in enumerate(np.diff(mesh)):
        endpoint, _ = local_flow(
            x[:, k], thrust[:, k] / case.max_thrust_n, alpha[k], case.tof_nd * fraction
        )
        defects[:, k] = np.asarray(endpoint).ravel() - x[:, k + 1]
    relative = np.maximum.reduce(
        [
            np.linalg.norm(defects[:3], axis=0)
            / np.maximum(
                np.maximum(np.linalg.norm(x[:3, :-1], axis=0), np.linalg.norm(x[:3, 1:], axis=0)),
                1e-13,
            ),
            np.linalg.norm(defects[3:6], axis=0)
            / np.maximum(
                np.maximum(np.linalg.norm(x[3:6, :-1], axis=0), np.linalg.norm(x[3:6, 1:], axis=0)),
                1e-13,
            ),
            abs(defects[6]) / np.maximum(np.maximum(abs(x[6, :-1]), abs(x[6, 1:])), 1e-13),
        ]
    )
    if not np.all(np.isfinite(relative)):
        raise FloatingPointError("Nonfinite local RK7 defects")
    return defects, relative


def integration_diagnostics(problem, arrays, directory, substeps=64):
    """Report local defects and full-transfer, physical-thrust propagation."""
    case, mesh = problem.case, arrays["mesh"]
    scale = problem.mission.scales[0].state
    x, thrust = arrays["states_nd"], arrays["thrust_n"]
    defects, relative = local_defects(problem, arrays, substeps)

    def describe(delta):
        physical = delta * scale[:, None]
        return dict(
            max_position_m=float(np.max(np.linalg.norm(physical[:3], axis=0))),
            max_velocity_m_s=float(np.max(np.linalg.norm(physical[3:6], axis=0))),
            max_mass_kg=float(np.max(abs(physical[6]))),
            final_position_m=float(np.linalg.norm(physical[:3, -1])),
            final_velocity_m_s=float(np.linalg.norm(physical[3:6, -1])),
            final_mass_kg=float(abs(physical[6, -1])),
            max_absolute_scaled_component=float(np.max(abs(delta))),
        )

    references = {}
    archive = dict(local_defects_nd=defects, local_relative_defects=relative)
    for name, rtol, position_atol, other_atol, steps in (
        ("tighter", 1e-13, 1e-7, 1e-11, 16),
        ("tightest", 3e-14, 1e-8, 1e-12, 32),
    ):
        states = np.empty_like(x)
        states[:, 0] = case.x0_augmented_state
        atol = np.r_[np.full(3, position_atol), np.full(4, other_atol)] / scale
        evaluations = 0
        minimum_moon_clearance = np.inf
        for k, fraction in enumerate(np.diff(mesh)):
            u = thrust[:, k] / case.thrust_unit
            sigma = np.linalg.norm(u)  # Actual physical thrust in full reintegration.
            dt = case.tof_nd * fraction
            answer = solve_ivp(
                lambda t, y: reference.eom(case, y, u, sigma),
                (0.0, dt),
                states[:, k],
                method="DOP853",
                rtol=rtol,
                atol=atol,
                max_step=dt / steps,
            )
            if not answer.success:
                raise RuntimeError(f"{name} integration, arc {k}: {answer.message}")
            states[:, k + 1] = answer.y[:, -1]
            evaluations += answer.nfev
            if is_cr3bp(case):
                minimum_moon_clearance = min(
                    minimum_moon_clearance, float(moon_clearance_km(case, answer.y).min())
                )
        archive[name + "_states_nd"] = states
        references[name] = dict(
            rtol=rtol,
            atol_si=(atol * scale).tolist(),
            maximum_step_fraction=1.0 / steps,
            function_evaluations=evaluations,
            error_against_optimizer=describe(states - x),
        )
        if is_cr3bp(case):
            references[name]["sampled_minimum_moon_clearance_m"] = minimum_moon_clearance * 1000.0
            references[name][
                "moon_sampling"
            ] = "Accepted DOP853 time points; report only, not a certified continuous minimum"
    reference_difference = describe(archive["tighter_states_nd"] - archive["tightest_states_nd"])
    diagnostics = dict(
        report_only=True,
        used_for_acceptance=False,
        local=dict(
            integrator="RK7",
            substeps=substeps,
            mass_flow="Tmax * throttle, matching the optimization",
            criterion="max relative r/v/m error, scaled by max of both endpoint magnitudes with ND floor 1e-13",
            max_relative_defect=float(relative.max()),
            **describe(defects),
        ),
        continuous=dict(
            integrator="DOP853",
            mass_flow="norm of physical thrust",
            state_resets=False,
            comparisons=references,
            tighter_minus_tightest=reference_difference,
            note="Reference differences measure numerical sensitivity, not a rigorous error bound.",
        ),
    )
    np.savez_compressed(directory / "integration_diagnostics.npz", **archive)
    write_json(directory / "integration_diagnostics.json", diagnostics)
    return diagnostics


def run_case(
    case_id,
    objective,
    directory,
    options,
    initial_intervals=50,
    max_refinements=8,
    defect_tolerance=None,
    max_intervals=1800,
):
    case = reference.DETERMINISTIC_CASE_REGISTRY[case_id]()
    if defect_tolerance is None:
        defect_tolerance = default_defect_tolerance(case)
    if (
        initial_intervals < 2
        or max_refinements < 0
        or max_intervals < initial_intervals
        or not np.isfinite(defect_tolerance)
        or defect_tolerance <= 0.0
    ):
        raise ValueError("Invalid mesh refinement configuration")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    problem = ShootingProblem(case, objective, options, method="hs", intervals=initial_intervals)
    checks = flow_checks(problem)
    write_json(directory / "flow_checks.json", checks)
    for check in checks:
        if (
            check["jacobian_error"] > 2e-6
            or check["hessian_vector_error"] > 2e-5
            or check["manual_rk7_difference_nd"] > 1e-12
        ):
            raise ValueError(f"Flow verification failed: {check}")
    primal = np.asarray(problem.opti.debug.value(problem.opti.x, problem.opti.initial())).ravel()
    np.savez_compressed(directory / "initial_guess.npz", **problem.seed, primal=primal)
    case = problem.case
    snapshot = directory / "source_snapshot"
    snapshot.mkdir()
    sources = ["deterministic_traj_opt.py", "integrator.py", "plotter.py"]
    sources += [p.as_posix() for p in Path("deterministic_optimization").glob("*.py")]
    sources += [p.as_posix() for p in Path("psyche_mission_events_reconstruction").glob("*.py")]
    hashes = {}
    for source in sources:
        destination = snapshot / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[source] = digest(destination)
    config = dict(
        case=case_id,
        objective=objective,
        initialization="fresh rollout -> HS -> multiple shooting",
        method="HS followed by multiple shooting",
        mass_flow_model="throttle_direction",
        initial_intervals=len(problem.mesh) - 1,
        shooting_substeps=problem.substeps,
        defect_substeps=64,
        moon_surface_nodes=problem.moon_surface_nodes,
        moon_radius_km=case.moon_radius if problem.moon_surface_nodes else None,
        moon_constraint_definition=(
            "squared Moon-centred distance / radius^2 >= 1 at all mesh boundaries"
            if problem.moon_surface_nodes
            else None
        ),
        rollout_substeps=2,
        rollout_thrust_per_component_n=1e-4,
        mesh_refinement=max_refinements > 0,
        max_refinements=max_refinements,
        max_intervals=max_intervals,
        defect_tolerance=defect_tolerance,
        refinement_policy="HS once, then bisect failing shooting arcs and re-solve; Hermite states, inherited throttle/direction",
        local_defect_criterion="max relative r/v/m error, scaled by max of both endpoint magnitudes with ND floor 1e-13",
        local_defects_used_for_acceptance=True,
        diagnostics_used_for_acceptance=False,
        global_diagnostics_used_for_acceptance=False,
        ipopt_options=options,
        initial_primal_sha256=hashlib.sha256(primal.tobytes()).hexdigest(),
        source_sha256=hashes,
        state_scales_si=problem.mission.scales[0].state.tolist(),
        time_scale_s=case.time_unit,
        time_of_flight_days=case.tof_days,
        initial_state_nd=case.x0_augmented_state.tolist(),
        final_state_nd=case.xf_state.tolist(),
        wet_mass_kg=case.m0_wet,
        dry_mass_kg=case.m0_dry,
        thrust_max_n=case.max_thrust_n,
        isp_s=case.isp,
        cost_scaling="fuel / wet mass; energy / (Tmax^2 * time of flight), as in Psyche",
        constraints="Original fixed states and duration; throttle, unit direction and mass floor."
        + (
            " Moon-surface exclusion at mesh boundaries only in HS and shooting."
            if problem.moon_surface_nodes
            else " No added surface inequalities."
        ),
    )
    write_json(directory / "configuration.json", config)
    hs_directory = directory / "hermite_simpson"
    hs_directory.mkdir()
    hs_result, hs_arrays = problem.solve(hs_directory)
    hs_result["stage_directory"] = hs_directory.name
    history = [hs_result]
    result, arrays = hs_result.copy(), hs_arrays
    last_directory = hs_directory
    defects, relative = None, None
    termination = "Hermite-Simpson did not converge; multiple shooting was not started"
    mesh = problem.mesh
    for refinement in range(max_refinements + 1 if hs_result["converged"] else 0):
        problem = ShootingProblem(
            case, objective, options, method="shooting", incoming=(result, arrays), mesh=mesh
        )
        initial = lambda expr: np.asarray(problem.opti.debug.value(expr, problem.opti.initial()))
        if refinement == 0:
            np.testing.assert_array_equal(initial(problem.X), hs_arrays["states_nd"])
            np.testing.assert_array_equal(initial(problem.D), hs_arrays["direction"])
            np.testing.assert_array_equal(initial(problem.A).ravel(), hs_arrays["throttle"])
            write_json(
                directory / "hs_to_shooting_handoff.json",
                dict(
                    verified=True,
                    states_identical=True,
                    throttle_identical=True,
                    direction_identical=True,
                    mesh_identical=True,
                    solver_settings_identical=True,
                    state_or_control_polishing=False,
                ),
            )
        last_directory = directory / (
            "shooting" if refinement == 0 else f"shooting_{refinement:02d}"
        )
        last_directory.mkdir()
        np.savez_compressed(
            last_directory / "initial_guess.npz",
            mesh=mesh,
            states_nd=initial(problem.X),
            direction=initial(problem.D),
            throttle=initial(problem.A).ravel(),
            primal=initial(problem.opti.x).ravel(),
        )
        write_json(
            last_directory / "initialization.json",
            dict(
                source=problem.initial_source,
                ipopt_options=options,
                refinement_index=refinement,
                parent_stage=history[-1]["stage_directory"],
            ),
        )
        print(f"SHOOTING mesh {refinement}: {len(mesh)-1} arcs", flush=True)
        result, arrays = problem.solve(last_directory)
        result["stage_directory"] = last_directory.name
        defects, relative = None, None
        if result["converged"]:
            try:
                defects, relative = local_defects(problem, arrays, 64)
                result["max_relative_local_defect"] = float(relative.max())
                result["local_defect_tolerance_met"] = bool(np.all(relative <= defect_tolerance))
                write_json(
                    last_directory / "local_defects.json",
                    dict(
                        substeps=64,
                        relative=relative,
                        defects_nd=defects,
                        tolerance=defect_tolerance,
                        criterion=config["local_defect_criterion"],
                        used_for_acceptance=True,
                    ),
                )
                print(f"Maximum local relative defect: {relative.max():.9g}", flush=True)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                result["local_defect_error"] = str(exc)
        write_json(last_directory / "result.json", result)
        history.append(result.copy())
        write_json(directory / "history.json", history)
        if not result["converged"]:
            termination = "Multiple-shooting NLP did not converge"
            break
        if relative is None:
            termination = "Local defect evaluation failed"
            break
        if np.all(relative <= defect_tolerance):
            termination = "Local defect tolerance reached"
            break
        if refinement == max_refinements:
            termination = "Mesh-refinement iteration limit reached"
            break
        mesh, selected = refine_mesh(mesh, relative, defect_tolerance, max_intervals)
        write_json(
            last_directory / "refinement.json",
            dict(
                split_indices=selected,
                next_mesh=mesh,
                next_arcs=len(mesh) - 1,
                criterion="local defects only",
            ),
        )
        if not selected:
            termination = "Mesh interval limit reached"
            break
    nlp_converged = bool(
        hs_result["converged"] and result["method"] == "shooting" and result["converged"]
    )
    defects_met = bool(relative is not None and np.all(relative <= defect_tolerance))
    result = dict(
        result,
        nlp_converged=nlp_converged,
        converged=bool(nlp_converged and defects_met),
        local_defect_tolerance_met=defects_met,
        termination_reason=termination,
        hs_converged=hs_result["converged"],
        shooting_started=hs_result["converged"],
        history=history,
        mesh_refinements_performed=max(0, len(history) - 2),
        initial_arcs=initial_intervals,
        combined_solver_seconds=sum(stage["solver_seconds"] for stage in history),
        last_stage_directory=str(last_directory.resolve()),
    )
    shutil.copyfile(last_directory / "solution.npz", directory / "solution.npz")
    try:
        diagnostics = integration_diagnostics(problem, arrays, directory)
        result["max_relative_local_defect"] = diagnostics["local"]["max_relative_defect"]
        result["continuous_physical_thrust_errors"] = diagnostics["continuous"]["comparisons"][
            "tightest"
        ]["error_against_optimizer"]
        result["continuous_reference_difference"] = diagnostics["continuous"][
            "tighter_minus_tightest"
        ]
        if problem.moon_surface_nodes:
            result["continuous_sampled_minimum_moon_clearance_m"] = diagnostics["continuous"][
                "comparisons"
            ]["tightest"]["sampled_minimum_moon_clearance_m"]
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        result["diagnostic_error"] = str(exc)
    result["total_seconds"] = time.perf_counter() - started
    write_json(directory / "result.json", result)
    from .plotting import plot_single_phase

    plot_single_phase(problem, arrays, directory, result)
    print(json.dumps(result, indent=2), flush=True)
    return result
