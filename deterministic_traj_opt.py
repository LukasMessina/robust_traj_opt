"""Deterministic low-thrust optimisation for CR3BP, Earth-Mars and Psyche.

    python deterministic_traj_opt.py --case all --objective both --jobs 3

Every objective starts independently with a dynamics rollout, followed by
Hermite-Simpson and h-adaptive RK7 multiple shooting. Controls are throttle
and unit direction. Energy means 0.5 * integral(dot(T,T) dt), in N^2 s.

Case classes and dynamics remain importable here for stochastic consumers.
See DETERMINISTIC_OPTIMISATION.md and psyche_mission_events_reconstruction/README.md.
"""

from deterministic_optimization.models import (
    SpacecraftScaling,
    CR3BPEarthMoon,
    TestCase,
    NrhoL2ToDro,
    LyapunovL1ToL2,
    HaloL2ToHaloL1,
    EarthMars,
    CASE_TYPES,
    CASE_REGISTRY,
    DETERMINISTIC_CASE_REGISTRY,
    earth_mars_data,
    eom,
    get_collinear_lagrange_points,
    propagate_periodic_orbit,
    rollout_initial_guess,
    integrate_with_dense_control,
)


def main(argv=None):
    from deterministic_optimization.cli import main as run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
