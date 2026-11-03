"""Linear dynamics, diffusion calibration and prescribed ShARK/Tsit5 integration."""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm
from diffrax import (
    AbstractPath,
    ControlTerm,
    ODETerm,
    MultiTerm,
    SaveAt,
    ConstantStepSize,
    SpaceTimeLevyArea,
    diffeqsolve,
    ShARK,
    Tsit5,
)

if TYPE_CHECKING:
    from stochastic_trajectory_optimization.double_integrator import Options
NX, NU = 4, 2


def unscented_weights(kappa: float, dimension: int) -> np.ndarray:
    weights = np.full(2 * dimension + 1, 1.0 / (2.0 * (dimension + kappa)))
    weights[0] = kappa / (dimension + kappa)
    return weights


def discrete_time_matrices(
    options: Options,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = options.dt
    A = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    B = np.array(
        [
            [0.5 * dt**2, 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt],
        ]
    )
    G = 0.01 * np.eye(NX)
    return A, B, G


def continuous_time_matrices() -> tuple[np.ndarray, np.ndarray]:
    Ac = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    Bc = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    return Ac, Bc


def accumulated_process_covariance(
    Ac: np.ndarray, Qc: np.ndarray, duration: float
) -> np.ndarray:
    """Compute the accumulated continuous-time process noise
    state_covariance by numerical quadrature to verify equivalence with the discrete noise model."""

    state_covariance, _ = quad_vec(
        lambda s: expm(Ac * s) @ Qc @ expm(Ac.T * s),
        0.0,
        duration,
        epsabs=1e-14,
        epsrel=1e-14,
    )
    return 0.5 * (state_covariance + state_covariance.T)


def calibrate_continuous_diffusion(
    Ac: np.ndarray, discrete_covariance: np.ndarray, duration: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover continuous time process noise state_covariance matrix Qc and its
    cholesky factor from a desired arc state_covariance.

    The finite-horizon continuous Lyapunov operator is formed column-by-column,
    then inverted. This deliberately calibrates the physical continuous SDE.
    """

    dimension = Ac.shape[0]
    lyapunov_operator = np.empty((dimension**2, dimension**2))
    for column in range(dimension**2):
        basis = np.zeros((dimension, dimension))
        basis.flat[column] = 1.0
        image, _ = quad_vec(
            lambda s: expm(Ac * s) @ basis @ expm(Ac.T * s),
            0.0,
            duration,
            epsabs=1e-14,
            epsrel=1e-14,
        )
        lyapunov_operator[:, column] = image.reshape(-1)
    Qc = np.linalg.solve(lyapunov_operator, discrete_covariance.reshape(-1)).reshape(
        dimension, dimension
    )
    Qc = 0.5 * (Qc + Qc.T)
    return Qc, np.linalg.cholesky(Qc)


def _drift_vector_field(t, state, control):
    Ac, Bc = continuous_time_matrices()
    return jnp.asarray(Ac) @ state + jnp.asarray(Bc) @ control


_TERM = ODETerm(lambda t, y, args: _drift_vector_field(t, y, args))


_SOLVER = Tsit5()


def diffrax_step(state: jnp.ndarray, control: jnp.ndarray, step: float) -> jnp.ndarray:
    """Propagate ``state`` one arc of duration ``step`` under zero-order-held ``control``.

    A single fixed Tsit5 step (``ConstantStepSize``) is exact for this linear,
    ZOH system regardless of solver order, so no substepping or adaptive
    tolerance is needed -- this differs from the reference repo's use of
    diffrax only in that the nonlinear CR3BP/2BP dynamics there require
    genuine adaptive integration, while here diffrax is used for the same
    linear dynamics the CasADi version integrated in closed form.
    """

    solution = diffeqsolve(
        _TERM,
        _SOLVER,
        t0=0.0,
        t1=step,
        dt0=step,
        y0=state,
        args=control,
        stepsize_controller=ConstantStepSize(),
        saveat=SaveAt(t1=True),
        max_steps=4,
    )
    return solution.ys[0]


_diffrax_step_batch = jax.vmap(diffrax_step, in_axes=(1, 1, None), out_axes=1)


def _diffusion_vector_field(t, state, Lc):
    del t, state
    return Lc


_SDE_SOLVER = ShARK()


class PrescribedSpaceTimeLevyPath(AbstractPath):
    """A deterministic one-arc Brownian driver carrying prescribed ``W`` and ``H``.

    ShARK requests the control over the complete integration step with
    ``use_levy=True``. The path therefore exposes both the Brownian increment
    ``W`` and Diffrax's normalized space-time Levy area ``H``. The linear
    interpolation below only defines the ordinary path value outside that
    full-step request; no Brownian subincrements are reconstructed from the
    two supplied random variables.
    """

    brownian_increment: jnp.ndarray
    space_time_levy_area: jnp.ndarray
    duration: float

    @property
    def t0(self):
        return 0.0

    @property
    def t1(self):
        return self.duration

    def evaluate(self, t0, t1=None, left=True, use_levy=False):
        del left
        if t1 is None:
            t1 = t0
            t0 = 0.0

        interval = t1 - t0
        fraction = interval / self.duration
        increment = fraction * self.brownian_increment
        if use_levy:
            return SpaceTimeLevyArea(
                dt=interval,
                W=increment,
                H=fraction * self.space_time_levy_area,
            )
        return increment


def prescribed_integration_sde_step(
    state: jnp.ndarray,
    control: jnp.ndarray,
    brownian_increment: jnp.ndarray,
    space_time_levy_area: jnp.ndarray,
    Lc: jnp.ndarray,
    step: float,
) -> jnp.ndarray:
    """Take one ShARK step using exactly the supplied ``W`` and ``H``.

    The prescribed path contains no random key, so the integration cannot
    introduce another Brownian realization during optimization or automatic
    differentiation. For one interval of length ``step``, callers supply
    independent standard-normal coordinates ``xi`` and ``eta`` through
    ``W = sqrt(step) xi`` and ``H = sqrt(step / 12) eta``.
    """

    prescribed_path = PrescribedSpaceTimeLevyPath(
        brownian_increment=brownian_increment,
        space_time_levy_area=space_time_levy_area,
        duration=step,
    )
    terms = MultiTerm(
        ODETerm(lambda t, y, args: _drift_vector_field(t, y, args[0])),
        ControlTerm(
            lambda t, y, args: _diffusion_vector_field(t, y, args[1]), prescribed_path
        ),
    )
    solution = diffeqsolve(
        terms,
        _SDE_SOLVER,
        t0=0.0,
        t1=step,
        dt0=step,
        y0=state,
        args=(control, Lc),
        stepsize_controller=ConstantStepSize(),
        saveat=SaveAt(t1=True),
        max_steps=1,
    )
    return solution.ys[0]


_prescribed_sde_integration_step_batch = jax.vmap(
    prescribed_integration_sde_step,
    in_axes=(1, 1, 1, 1, None, None),
    out_axes=1,
)
