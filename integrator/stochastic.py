"""JAX deterministic and prescribed-noise trajectory integration maps."""

from __future__ import annotations
import jax
import jax.numpy as jnp
import diffrax
import numpy as np
from .rk9 import XMDS2RK9, XMDS2_RK9_A, XMDS2_RK9_B_SOL, XMDS2_RK9_C


def step_adjoint():
    """Reverse-mode adjoint of the fixed-step Diffrax solves.

    Both choices differentiate the same discrete RK steps exactly. Split over
    several devices, DirectAdjoint is required: the Equinox checkpointed while
    loop behind RecursiveCheckpointAdjoint does not trace under shard_map.
    """
    if jax.device_count() > 1:
        return diffrax.DirectAdjoint()
    return diffrax.RecursiveCheckpointAdjoint()


def vector_field(case, state, control, magnitude):
    """Shared physical scales; heliocentric mu=1 or rotating-frame CR3BP."""
    if getattr(case, "dynamics_model", "cr3bp") == "heliocentric":
        position = state[:3]
        acceleration = -position / jnp.linalg.norm(position) ** 3 + control / state[6]
        return jnp.concatenate(
            (
                state[3:6],
                acceleration,
                jnp.atleast_1d(-magnitude / case.exhaust_velocity_nd),
            )
        )
    x, y, z, vx, vy, vz, mass = state
    r1 = jnp.sqrt((x + case.mu) ** 2 + y * y + z * z)
    r2 = jnp.sqrt((x - 1 + case.mu) ** 2 + y * y + z * z)
    gravity = (1 - case.mu) / r1**3 + case.mu / r2**3
    return jnp.stack(
        (
            vx,
            vy,
            vz,
            2 * vy
            + x
            - (1 - case.mu) * (x + case.mu) / r1**3
            - case.mu * (x - 1 + case.mu) / r2**3
            + control[0] / mass,
            -2 * vx + y - gravity * y + control[1] / mass,
            -gravity * z + control[2] / mass,
            -magnitude / case.exhaust_velocity_nd,
        )
    )


def integration_map(case, substeps):
    """Fixed-substep Diffrax Dopri8 (order 8), differentiated through its steps.

    Integrating on [0,1] avoids a final tiny step from roundoff in arc times.
    Reverse-mode AD uses Diffrax's discretize-then-optimize checkpoint adjoint.
    No adaptive tolerance or stochastic differential equation changes the model.
    """
    if int(substeps) != substeps or substeps < 1:
        raise ValueError("integrator_substeps must be a positive integer")
    substeps = int(substeps)

    def rhs(t, state, args):
        control, magnitude, duration = args
        return duration * vector_field(case, state, control, magnitude)

    term, solver = diffrax.ODETerm(rhs), diffrax.Dopri8()

    def step(state, control, magnitude, duration):
        result = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=1.0 / substeps,
            y0=state,
            args=(control, magnitude, duration),
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=substeps + 1,
            adjoint=step_adjoint(),
        )
        return result.ys[0]

    return step


class PrescribedWienerPath(diffrax.AbstractPath):
    """Linear path carrying one externally prescribed Wiener increment.

    XMDS2 holds the sampled Wiener rate constant over a fixed RK step. Hence
    an interval occupying a fraction of the step receives the same fraction
    of ``brownian_increment``. No random sampling occurs inside this path.
    """

    brownian_increment: jnp.ndarray
    duration: float

    @property
    def t0(self):
        return 0.0

    @property
    def t1(self):
        return self.duration

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is None:
            t1 = t0
            t0 = 0.0
        interval = t1 - t0
        fraction = interval / self.duration
        return fraction * self.brownian_increment


def stochastic_integration_map(case, options, integration_refinement=1):
    """Return a fixed-grid XMDS2 RK9 map for the additive CR3BP SDE.

    ``brownian_increments`` has shape ``(M, 3)``, where ``M`` is
    ``sde_integrator_substeps``. Each substep is one fixed RK9 step with a
    constant Wiener rate, represented by the prescribed path's total
    ``delta W``. The thrust command and its mass-flow magnitude remain held
    over the complete shooting arc.

    The returned map carries an exact derivative rule: one sigma-point step
    has NX outputs, so its full Jacobian costs NX reverse sweeps, after which
    every tangent or cotangent passes through it as a small matrix product.
    Reverse-mode Jacobians of the arc map then need NX sweeps per sigma point
    instead of one per arc output.
    """

    if (
        int(integration_refinement) != integration_refinement
        or integration_refinement < 1
    ):
        raise ValueError("integration_refinement must be a positive integer")
    substeps = int(options.sde_integrator_substeps)
    if substeps != options.sde_integrator_substeps or substeps < 1:
        raise ValueError("sde_integrator_substeps must be a positive integer")
    from stochastic_trajectory_optimization.dynamics import (
        acceleration_diffusion_matrix,
    )

    diffusion = jnp.asarray(acceleration_diffusion_matrix(case, options))
    drift_term = diffrax.ODETerm(
        lambda t, state, args: vector_field(case, state, args[0], args[1])
    )
    solver = XMDS2RK9()

    def one_substep(state, brownian_increment, control, magnitude, substep_duration):
        path = PrescribedWienerPath(
            brownian_increment=brownian_increment,
            duration=substep_duration,
        )
        diffusion_term = diffrax.ControlTerm(
            lambda t, state, args: diffusion,
            path,
        )
        result = diffrax.diffeqsolve(
            diffrax.MultiTerm(drift_term, diffusion_term),
            solver,
            t0=0.0,
            t1=substep_duration,
            dt0=substep_duration,
            y0=state,
            args=(control, magnitude),
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=1,
            adjoint=step_adjoint(),
        )
        return result.ys[0]

    def scan_step(
        current_state, brownian_increment, control, magnitude, substep_duration
    ):
        if integration_refinement > 1:
            # Diagnostic refinement of the SAME prescribed Wiener path and UT
            # ensemble, without changing the stochastic discretisation.
            next_state = jax.lax.fori_loop(
                0,
                integration_refinement,
                lambda _, y: one_substep(
                    y,
                    brownian_increment / integration_refinement,
                    control,
                    magnitude,
                    substep_duration / integration_refinement,
                ),
                current_state,
            )
            return next_state, next_state
        next_state = one_substep(
            current_state,
            brownian_increment,
            control,
            magnitude,
            substep_duration,
        )
        return next_state, next_state

    def step(
        state,
        control,
        magnitude,
        brownian_increments,
        duration,
    ):
        substep_duration = duration / substeps
        final_state, _ = jax.lax.scan(
            lambda current_state, brownian_increment: scan_step(
                current_state, brownian_increment, control, magnitude, substep_duration
            ),
            state,
            brownian_increments,
        )
        return final_state

    def step_with_history(
        state,
        control,
        magnitude,
        brownian_increments,
        duration,
    ):
        """Like `step`, but also returns every intermediate substep state.

        Returns ``(final_state, substep_states)`` with ``substep_states`` of
        shape ``(sde_integrator_substeps, NX)``, the state after each of the
        ``substeps`` RK9 stages (the last row equals ``final_state``). Display
        use only: propagation, effort accounting and moment matching all use
        `step`, which discards these intermediate states.
        """

        substep_duration = duration / substeps
        final_state, substep_states = jax.lax.scan(
            lambda current_state, brownian_increment: scan_step(
                current_state, brownian_increment, control, magnitude, substep_duration
            ),
            state,
            brownian_increments,
        )
        return final_state, substep_states

    @jax.custom_jvp
    def step_with_jacobian(state, control, magnitude, brownian_increments, duration):
        return step(state, control, magnitude, brownian_increments, duration)

    @step_with_jacobian.defjvp
    def step_jvp(primals, tangents):
        # Rows of the step Jacobian with respect to every input (state, control,
        # magnitude, Brownian increments, duration), from NX reverse sweeps.
        final_state, pullback = jax.vjp(step, *primals)
        rows = jax.vmap(pullback)(jnp.eye(final_state.shape[0]))
        return final_state, sum(
            jnp.tensordot(row, tangent, axes=jnp.ndim(tangent))
            for row, tangent in zip(rows, tangents)
        )

    step_with_jacobian.with_history = step_with_history
    return step_with_jacobian


class Dynamics:
    """Diffrax maps and batched exact discrete derivatives for seeds/reporting."""

    def __init__(self, case):
        self.case = case
        self._cache = {}

    def _maps(self, substeps, *, batched_steps=False):
        key = (substeps, batched_steps)
        axes = (1, 1, 0 if batched_steps else None)
        if key not in self._cache:
            step = integration_map(self.case, substeps)

            def raw(state, control, duration):
                return step(state, control, jnp.linalg.norm(control), duration)

            self._cache[key] = (
                jax.jit(jax.vmap(raw, in_axes=axes, out_axes=1)),
                jax.jit(jax.vmap(jax.jacrev(raw, argnums=(0, 1)), in_axes=axes)),
            )
        return self._cache[key]

    def propagate(self, states, controls, step, substeps, with_jacobian=False):
        propagation, derivatives = self._maps(substeps)
        result = np.asarray(propagation(states, controls, step))
        if with_jacobian:
            a, b = derivatives(states, controls, step)
            return result, np.asarray(a), np.asarray(b)
        return result, None, None


def nominal_map(case, substeps, eps, save_substeps=False):
    def rhs(t, x, args):
        command, dt = args
        magnitude = case.max_thrust_nd * jnp.sqrt(command @ command + eps**2)
        return dt * vector_field(case, x, case.max_thrust_nd * command, magnitude)

    term, solver = diffrax.ODETerm(rhs), XMDS2RK9()

    def step(state, command, duration):
        result = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=1 / substeps,
            y0=state,
            args=(command, duration),
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=(
                diffrax.SaveAt(ts=jnp.arange(1, substeps + 1) / substeps)
                if save_substeps
                else diffrax.SaveAt(t1=True)
            ),
            max_steps=substeps + 1,
            adjoint=step_adjoint(),
        )
        return result.ys if save_substeps else result.ys[0]

    return step


def ensemble_integrator(data, sample_specific=True):
    """Fixed RK9 steps with bracketed SOI and closest-approach events.

    Every event subinterval uses the SAME command and Wiener rate. A radial
    minimum is checked when both SOI endpoint distances are positive, so a
    complete encounter inside one coarse cruise step is not simply skipped.
    """
    from stochastic_trajectory_optimization.dynamics import split_over_devices

    mission = data.mission
    mu_sun = mission.scales[0].mu
    mu_mars = mission.scales[1].mu
    soi = mission.soi[499]
    planet_batch = jax.vmap(data.planet)

    def relative(y, et):
        return y[:, :6] - planet_batch(et)

    def step_one(y, force, rate, et, dt, mode):
        def rhs(x, t):
            planet, planet_rate = jax.jvp(data.planet, (t,), (jnp.array(1.0),))
            r = x[:3] - planet[:3]
            solar = -mu_sun * x[:3] / jnp.linalg.norm(x[:3]) ** 3
            martian = -mu_mars * r / jnp.linalg.norm(r) ** 3 + planet_rate[3:6]
            # The derivative of the actual position spline need not equal its
            # independently fitted velocity. This is the exact affine reset rate.
            velocity = x[3:6] + jnp.where(mode == 1, planet_rate[:3] - planet[3:6], 0.0)
            acceleration = jnp.where(mode == 1, martian, solar) + force / x[6] + rate
            # Exact thrust norm (a simulation: no derivative is taken), as in the other
            # cases' Monte Carlo; an engine-off (ballistic Mars) command is exactly zero
            # and consumes no propellant.
            thrust = force @ force
            flow = jnp.sqrt(thrust)
            mass = -flow / (mission.settings.isp_s * 9.80665)
            return jnp.concatenate((velocity, acceleration, jnp.atleast_1d(mass)))

        stages = []
        for i in range(16):
            value = y
            for k in range(i):
                if XMDS2_RK9_A[i, k]:
                    value = value + dt * XMDS2_RK9_A[i, k] * stages[k]
            stages.append(rhs(value, et + dt * XMDS2_RK9_C[i]))
        result = y + dt * sum(
            XMDS2_RK9_B_SOL[i] * stages[i] for i in range(16) if XMDS2_RK9_B_SOL[i]
        )
        return result

    flow = jax.vmap(step_one)

    def radial(y, et):
        z = relative(y, et)
        return jnp.sum(z[:, :3] * z[:, 3:6], axis=1)

    def surface(y, et):
        return jnp.linalg.norm(relative(y, et)[:, :3], axis=1) - soi

    def root(y, force, rate, et, mode, hi, kind):
        # kind 0: inward SOI; 1: outward SOI; 2: periapsis.
        lo = jnp.zeros_like(hi)

        def body(_, bounds):
            low, high = bounds
            mid = (low + high) / 2
            value = flow(y, force, rate, et, mid, mode)
            g = jax.lax.cond(
                kind == 2,
                lambda _: radial(value, et + mid),
                lambda _: surface(value, et + mid),
                None,
            )
            before = jnp.where(kind == 0, g > 0, g < 0)
            return jnp.where(before, mid, low), jnp.where(before, high, mid)

        lo, hi = jax.lax.fori_loop(0, 38, body, (lo, hi))
        return (lo + hi) / 2

    def record_minimum(y0, y1, force, rate, et, dt, mode, best_r, best_t, best_y):
        z0, z1 = relative(y0, et), relative(y1, et + dt)
        r0, r1 = jnp.linalg.norm(z0[:, :3], axis=1), jnp.linalg.norm(z1[:, :3], axis=1)
        take = r0 < best_r
        best_r, best_t, best_y = (
            jnp.minimum(best_r, r0),
            jnp.where(take, et, best_t),
            jnp.where(take[:, None], y0, best_y),
        )
        take = r1 < best_r
        best_r, best_t, best_y = (
            jnp.minimum(best_r, r1),
            jnp.where(take, et + dt, best_t),
            jnp.where(take[:, None], y1, best_y),
        )
        mask = (
            (jnp.sum(z0[:, :3] * z0[:, 3:6], axis=1) < 0)
            & (jnp.sum(z1[:, :3] * z1[:, 3:6], axis=1) >= 0)
            & (dt > 0)
        )

        def locate(carry):
            br, bt, by = carry
            tau = root(
                y0, force, rate, et, mode, jnp.where(mask, dt, 0.0), jnp.array(2)
            )
            y = flow(y0, force, rate, et, tau, mode)
            r = jnp.linalg.norm(relative(y, et + tau)[:, :3], axis=1)
            take = mask & (r < br)
            return (
                jnp.where(take, r, br),
                jnp.where(take, et + tau, bt),
                jnp.where(take[:, None], y, by),
            )

        return jax.lax.cond(
            jnp.any(mask), locate, lambda carry: carry, (best_r, best_t, best_y)
        )

    def advance(
        y,
        force,
        brownian,
        et_scalar,
        duration,
        mode,
        best_r,
        best_t,
        best_y,
        entry,
        exit,
    ):
        et = jnp.full(y.shape[0], et_scalar)
        dt = jnp.full(y.shape[0], duration)
        rate = 1e-7 * brownian / duration
        if not sample_specific:
            trial = flow(y, force, rate, et, dt, mode)
            best_r, best_t, best_y = record_minimum(
                y, trial, force, rate, et, dt, mode, best_r, best_t, best_y
            )
            return trial, mode, best_r, best_t, best_y, entry, exit
        # At most one entry and one exit in a macrostep. Mode 2 remains solar.
        for _ in range(2):
            trial = flow(y, force, rate, et, dt, mode)
            g1 = surface(trial, et + dt)
            entering = (mode == 0) & (g1 <= 0)
            leaving = (mode == 1) & (g1 >= 0)
            bracket = dt
            hidden = (
                (mode == 0)
                & (~entering)
                & (radial(y, et) < 0)
                & (radial(trial, et + dt) > 0)
            )

            def detect(args):
                entering, bracket = args
                closest = root(
                    y, force, rate, et, mode, jnp.where(hidden, dt, 0.0), jnp.array(2)
                )
                inside = (
                    surface(flow(y, force, rate, et, closest, mode), et + closest) < 0
                )
                found = hidden & inside
                return entering | found, jnp.where(found, closest, bracket)

            entering, bracket = jax.lax.cond(
                jnp.any(hidden), detect, lambda x: x, (entering, bracket)
            )
            crossing = (entering | leaving) & (dt > 0)

            def crossed(args):
                y, et, dt, mode, br, bt, by, entry, exit = args
                tin = root(
                    y,
                    force,
                    rate,
                    et,
                    mode,
                    jnp.where(entering, bracket, 0.0),
                    jnp.array(0),
                )
                tout = root(
                    y, force, rate, et, mode, jnp.where(leaving, dt, 0.0), jnp.array(1)
                )
                elapsed = jnp.where(entering, tin, jnp.where(leaving, tout, dt))
                at = flow(y, force, rate, et, elapsed, mode)
                br, bt, by = record_minimum(
                    y, at, force, rate, et, elapsed, mode, br, bt, by
                )
                entry = jnp.where(entering, et + elapsed, entry)
                exit = jnp.where(leaving, et + elapsed, exit)
                return (
                    at,
                    et + elapsed,
                    dt - elapsed,
                    jnp.where(crossing, mode + 1, mode),
                    br,
                    bt,
                    by,
                    entry,
                    exit,
                )

            def no_cross(args):
                y, et, dt, mode, br, bt, by, entry, exit = args
                br, bt, by = record_minimum(
                    y, trial, force, rate, et, dt, mode, br, bt, by
                )
                return trial, et + dt, jnp.zeros_like(dt), mode, br, bt, by, entry, exit

            y, et, dt, mode, best_r, best_t, best_y, entry, exit = jax.lax.cond(
                jnp.any(crossing),
                crossed,
                no_cross,
                (y, et, dt, mode, best_r, best_t, best_y, entry, exit),
            )
        # Finish any post-exit solar remainder, without another noise draw.
        trial = flow(y, force, rate, et, dt, mode)
        best_r, best_t, best_y = record_minimum(
            y, trial, force, rate, et, dt, mode, best_r, best_t, best_y
        )
        return trial, mode, best_r, best_t, best_y, entry, exit

    # Samples are independent: split the batch over every local device.
    return jax.jit(
        split_over_devices(advance, (True, True, True, False, False) + (True,) * 6)
    )
