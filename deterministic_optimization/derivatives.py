"""Independent accuracy and derivative checks for the shooting arc maps."""

import casadi as ca
import numpy as np
from .psyche_validation import error_record, propagate_interval, relative_interval_error
from .flows import get_arc_flow


def seed_solution(mission):
    phases = []
    for i in range(1, 5):
        seed = mission.seed
        phases.append(
            {
                "mesh": seed[f"phase{i}_tau"],
                "states_si": seed[f"phase{i}_states_si"],
                "thrust_n": seed[f"phase{i}_thrust_n"],
                "sigma_n": seed[f"phase{i}_sigma_n"],
                "duration_s": seed[f"phase{i}_elapsed_s"][-1],
                "start_et": seed[f"phase{i}_epochs_et_tdb_s"][0],
            }
        )
    return {"phases": phases, "exit_days": mission.seed_exit_days}


def integration_checks(mission, solution, substeps):
    """Compare every arc map to independent DOP853 from the SAME arc start.

    Integration error and NLP continuity error are separate quantities. Neither
    this report nor the continuous propagation changes the h-adaptation rule.
    """
    cfg, records = (mission.settings, [])
    for i, phase in enumerate(solution["phases"]):
        flow, scale = (get_arc_flow(mission, i, substeps), mission.scales[i])
        relative, errors, continuity = ([], [], [])
        for k, fraction in enumerate(np.diff(phase["mesh"])):
            dt = phase["duration_s"] * fraction
            x, thrust, sigma = (phase["states_si"][k], phase["thrust_n"][k], phase["sigma_n"][k])
            computed, _ = flow(
                x / scale.state,
                thrust / cfg.thrust_max_n,
                sigma / cfg.thrust_max_n,
                dt / scale.time,
            )
            computed = np.asarray(computed).ravel() * scale.state
            reference = propagate_interval(mission, i, x, thrust, sigma, dt).y[:, -1]
            delta = error_record(computed - reference)
            error = [delta["position_m"], delta["velocity_m_s"], delta["mass_kg"]]
            errors.append(error)
            relative.append(relative_interval_error(x, reference, error, scale))
            continuity.append(error_record(reference - phase["states_si"][k + 1]))
        errors = np.asarray(errors)
        records.append(
            {
                "phase": i + 1,
                "maximum_relative_integration_error": float(max(relative)),
                "max_position_integration_error_m": float(errors[:, 0].max()),
                "max_velocity_integration_error_m_s": float(errors[:, 1].max()),
                "relative_integration_errors": relative,
                "dop853_endpoint_mismatches": continuity,
            }
        )
    return {
        "method": "RK7 shooting map versus independent DOP853, identical arc starts/controls/sigma",
        "substeps": substeps,
        "additional_steps": 0,
        "integration_grid": "uniform",
        "maximum_relative_integration_error": max(
            (r["maximum_relative_integration_error"] for r in records)
        ),
        "phases": records,
    }


def derivative_checks(mission, substeps):
    records = []
    for i, scale in enumerate(mission.scales):
        flow = get_arc_flow(mission, i, substeps)
        z = ca.MX.sym("z", 12)
        endpoint, _ = flow(z[:7], z[7:10], z[10], z[11])
        jac = ca.jacobian(endpoint, z)
        jacobian = ca.Function("flow_jac", [z], [jac])
        weights = ca.DM([0.2, -0.3, 0.1, 0.4, -0.2, 0.5, 0.7])
        hessian = ca.Function("flow_hess", [z], [ca.hessian(ca.dot(weights, endpoint), z)[0]])
        values = ca.Function("flow_value", [z], [endpoint])
        states = mission.seed[f"phase{i + 1}_states_si"]
        indices = sorted(
            {0, min(int(np.argmin(np.linalg.norm(states[:, :3], axis=1))), len(states) - 2)}
        )
        for k in indices:
            dt = np.diff(mission.seed[f"phase{i + 1}_elapsed_s"])[k] / scale.time
            point = np.r_[states[k] / scale.state, [0.1, -0.15, 0.2], 0.3, dt]
            analytic = np.asarray(jacobian(point))
            numerical = np.empty_like(analytic)
            for j in range(12):
                shift = np.zeros(12)
                shift[j] = 2e-06 * max(abs(point[j]), 1.0)
                numerical[:, j] = np.asarray(
                    (values(point + shift) - values(point - shift)) / (2 * shift[j])
                ).ravel()
            jac_error = float(np.max(abs(analytic - numerical) / (1.0 + abs(analytic))))
            direction = np.linspace(-0.6, 0.7, 12)
            eps = 1e-05
            finite_hv = (
                np.asarray(
                    (jacobian(point + eps * direction) - jacobian(point - eps * direction))
                    / (2 * eps)
                ).T
                @ np.asarray(weights).ravel()
            )
            analytic_hv = np.asarray(hessian(point)) @ direction
            hess_error = float(np.max(abs(analytic_hv - finite_hv) / (1.0 + abs(analytic_hv))))
            records.append(
                {
                    "phase": i + 1,
                    "seed_interval": k,
                    "jacobian_normalised_error": jac_error,
                    "hessian_vector_normalised_error": hess_error,
                }
            )
    return {
        "checks": records,
        "maximum_jacobian_error": max((r["jacobian_normalised_error"] for r in records)),
        "maximum_hessian_vector_error": max(
            (r["hessian_vector_normalised_error"] for r in records)
        ),
        "covers": "State, thrust, sigma and arc duration; central finite differences versus exact CasADi AD",
    }


def exit_time_derivative_checks(problem):
    op = problem.opti
    initial = np.asarray(op.debug.value(op.x, op.initial())).ravel()
    values = ca.Function("all_constraints", [op.x], [op.g])
    jacobian = ca.Function("exit_time_jac", [op.x], [ca.jacobian(op.g, op.x)[:, :2]])
    analytic = np.asarray(jacobian(initial))
    errors = []
    for i in range(2):
        shift = np.zeros_like(initial)
        shift[i] = 1e-05
        numerical = np.asarray((values(initial + shift) - values(initial - shift)) / 2e-05).ravel()
        errors.append(float(np.max(abs(numerical - analytic[:, i]) / (1.0 + abs(analytic[:, i])))))
    return {
        "earth_exit_duration_error": errors[0],
        "mars_exit_duration_error": errors[1],
        "covers": "All constraints, including integrated dynamics, surface samples and ephemeris frame links",
    }


def reintegrate_sigma_model(mission, solution):
    state, records = (mission.x0.copy(), [])
    for i, phase in enumerate(solution["phases"]):
        for k, fraction in enumerate(np.diff(phase["mesh"])):
            sol = propagate_interval(
                mission,
                i,
                state,
                phase["thrust_n"][k],
                phase["sigma_n"][k],
                fraction * phase["duration_s"],
            )
            state = sol.y[:, -1].copy()
        records.append(
            {"phase": i + 1, "end_mismatch": error_record(state - phase["states_si"][-1])}
        )
        if i < 3:
            planet = mission.cache.state(
                399 if i == 0 else 499, phase["start_et"] + phase["duration_s"]
            )
            state[:6] += planet if i in (0, 2) else -planet
    return {
        "method": "Continuous DOP853 from TIP, no resets, sigma-based mass flow matching the NLP",
        "role": "Report only; never changes convergence/refinement",
        "phases": records,
    }


def control_derivative_checks(mission, solution, substeps):
    """Finite-difference checks of the composed alpha/direction shooting map.

    Include exactly zero throttle: its direction is unidentifiable but the
    map and its first/second derivatives remain defined.
    """
    rows = []
    for i, (phase, scale) in enumerate(zip(solution["phases"], mission.scales)):
        k = min(
            int(np.argmin(np.linalg.norm(phase["states_si"][:, :3], axis=1))),
            len(phase["mesh"]) - 2,
        )
        z = ca.MX.sym("z", 12)
        endpoint, _ = get_arc_flow(mission, i, substeps)(z[:7], z[7:10] * z[10], z[10], z[11])
        weights = ca.DM([0.2, -0.3, 0.1, 0.4, -0.2, 0.5, 0.7])
        fun = ca.Function("value", [z], [endpoint])
        jac = ca.Function("jacobian", [z], [ca.jacobian(endpoint, z)])
        hess = ca.Function("hessian", [z], [ca.hessian(ca.dot(weights, endpoint), z)[0]])
        for throttle in (0.0, 0.4):
            dt = phase["duration_s"] * np.diff(phase["mesh"])[k] / scale.time
            point = np.r_[phase["states_si"][k] / scale.state, [0.6, 0.0, 0.8], throttle, dt]
            analytic = np.asarray(jac(point))
            finite = np.empty_like(analytic)
            for j in range(12):
                shift = np.zeros(12)
                shift[j] = 2e-06 * max(abs(point[j]), 1.0)
                finite[:, j] = np.asarray(
                    (fun(point + shift) - fun(point - shift)) / (2 * shift[j])
                ).ravel()
            direction = np.linspace(-0.6, 0.7, 12)
            eps = 1e-05
            finite_hv = (
                np.asarray(
                    (jac(point + eps * direction) - jac(point - eps * direction)) / (2 * eps)
                ).T
                @ np.asarray(weights).ravel()
            )
            analytic_hv = np.asarray(hess(point)) @ direction
            rows.append(
                dict(
                    phase=i + 1,
                    interval=k,
                    throttle=throttle,
                    jacobian_error=float(np.max(abs(analytic - finite) / (1.0 + abs(analytic)))),
                    hessian_vector_error=float(
                        np.max(abs(analytic_hv - finite_hv) / (1.0 + abs(analytic_hv)))
                    ),
                )
            )
    return dict(
        checks=rows,
        maximum_jacobian_error=max((r["jacobian_error"] for r in rows)),
        maximum_hessian_vector_error=max((r["hessian_vector_error"] for r in rows)),
    )
