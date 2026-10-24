"""Check sensitivity of continuous propagation to the diagnostic integrator settings.

This never changes controls, epochs, the NLP, or its acceptance decision.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .psyche import Mission, Settings, load_solution
from .psyche_validation import error_record


def main(archive, output, candidate=None):
    archive = archive.resolve()
    config = json.loads((archive / "configuration.json").read_text(encoding="utf-8"))
    source = (
        str(Path(candidate).resolve())
        if candidate is not None
        else json.loads((archive / "result.json").read_text(encoding="utf-8"))["last_candidate"]
    )
    solution = load_solution(source)
    mission = Mission(Settings(**config["settings"]))
    variants = {
        "standard": dict(rtol=2e-12, atol=[1e-5] * 3 + [1e-9] * 4, step_factor=1.0),
        "tighter": dict(rtol=1e-13, atol=[1e-7] * 3 + [1e-11] * 4, step_factor=0.5),
        "tightest": dict(rtol=3e-14, atol=[1e-8] * 3 + [1e-12] * 4, step_factor=0.25),
    }
    records, endpoints = {}, {}
    for mass_model in ("sigma", "physical_thrust_norm"):
        records[mass_model], endpoints[mass_model] = {}, {}
        for name, settings in variants.items():
            state = mission.x0.copy()
            rows, states = [], []
            for i, phase in enumerate(solution["phases"]):
                for k, fraction in enumerate(np.diff(phase["mesh"])):
                    thrust = phase["thrust_n"][k]
                    sigma = phase["sigma_n"][k] if mass_model == "sigma" else np.linalg.norm(thrust)
                    dt = fraction * phase["duration_s"]
                    propagated = solve_ivp(
                        lambda t, y: mission.rhs_si(i, y, thrust, sigma),
                        (0.0, dt),
                        state,
                        method="DOP853",
                        rtol=settings["rtol"],
                        atol=settings["atol"],
                        max_step=settings["step_factor"]
                        * min(dt / 8.0, 86400.0 if i in (1, 3) else 900.0),
                    )
                    if not propagated.success:
                        raise RuntimeError(propagated.message)
                    state = propagated.y[:, -1].copy()
                states.append(state.copy())
                rows.append(
                    dict(phase=i + 1, end_mismatch=error_record(state - phase["states_si"][-1]))
                )
                if i < 3:
                    planet = mission.cache.state(
                        399 if i == 0 else 499, phase["start_et"] + phase["duration_s"]
                    )
                    state[:6] += planet if i in (0, 2) else -planet
            records[mass_model][name], endpoints[mass_model][name] = rows, states
        for name in ("standard", "tighter"):
            for i, row in enumerate(records[mass_model][name]):
                row["difference_from_tightest"] = error_record(
                    endpoints[mass_model][name][i] - endpoints[mass_model]["tightest"][i]
                )
    report = dict(
        source_candidate=source,
        method="Continuous DOP853 from TIP, no state resets",
        purpose="Report sensitivity to diagnostic integration settings; no new acceptance gate",
        qualification="Differences are convergence checks, not rigorous bounds on the exact continuous trajectory",
        integrator_settings=variants,
        mass_models=records,
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {model: {name: rows[-1] for name, rows in records[model].items()} for model in records},
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--candidate",
        type=Path,
        help="Optional specific saved iterate; otherwise use result.json last_candidate",
    )
    args = p.parse_args()
    main(args.archive, args.output, args.candidate)
