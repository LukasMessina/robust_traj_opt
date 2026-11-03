"""Independent diffusion and Sobol Monte Carlo checks of a double-integrator policy."""

from __future__ import annotations
from typing import TYPE_CHECKING
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm, qmc
from integrator.double_integrator import (
    NX,
    NU,
    unscented_weights,
    accumulated_process_covariance,
    _prescribed_sde_integration_step_batch,
)

if TYPE_CHECKING:
    from stochastic_trajectory_optimization.double_integrator import (
        Options,
        RobustTrajectory,
    )


def validate_shark_process_covariance(
    Ac: np.ndarray,
    Qc: np.ndarray,
    Lc: np.ndarray,
    duration: float,
) -> float:
    """Compare one ShARK step with the exact linear-SDE process covariance.

    A symmetric sigma rule integrates the covariance exactly because this
    double-integrator's one-step stochastic map is affine in the independent
    Gaussian variables defining ``W`` and ``H``.
    """

    noise_dimension = Lc.shape[1]
    stochastic_dimension = 2 * noise_dimension
    stochastic_coordinates = np.concatenate(
        [
            np.zeros((stochastic_dimension, 1)),
            np.sqrt(stochastic_dimension) * np.eye(stochastic_dimension),
            -np.sqrt(stochastic_dimension) * np.eye(stochastic_dimension),
        ],
        axis=1,
    )
    weights = unscented_weights(0.0, stochastic_dimension)
    brownian_increments = np.sqrt(duration) * stochastic_coordinates[:noise_dimension]
    space_time_levy_areas = (
        np.sqrt(duration / 12.0) * stochastic_coordinates[noise_dimension:]
    )
    n_sigma = stochastic_coordinates.shape[1]
    propagated = np.asarray(
        _prescribed_sde_integration_step_batch(
            jnp.zeros((NX, n_sigma)),
            jnp.zeros((NU, n_sigma)),
            jnp.asarray(brownian_increments),
            jnp.asarray(space_time_levy_areas),
            jnp.asarray(Lc),
            duration,
        )
    )
    propagated_mean = propagated @ weights
    residuals = propagated - propagated_mean[:, None]
    numerical_covariance = (residuals * weights[None, :]) @ residuals.T
    numerical_covariance = 0.5 * (numerical_covariance + numerical_covariance.T)
    exact_covariance = accumulated_process_covariance(Ac, Qc, duration)
    return float(np.max(np.abs(numerical_covariance - exact_covariance)))


def open_loop_state_covariances(
    options: Options,
    Lc: np.ndarray,
    solution: RobustTrajectory,
) -> np.ndarray:
    """Estimate node covariances from feedforward-only SDE rollouts."""

    return monte_carlo_rollout(options, Lc, solution, closed_loop=False)[
        "state_covariances"
    ]


def monte_carlo_rollout(
    options: Options,
    Lc: np.ndarray,
    solution: RobustTrajectory,
    *,
    closed_loop: bool = True,
) -> dict[str, np.ndarray]:
    """Roll out trajectories using independent scrambled Sobol input sets.

    Initial states and stochastic paths use separate random streams. Random
    pairing avoids alignment between the two Sobol sets. Each Brownian point
    supplies independent standard-normal coordinates for both ``W`` and ``H``
    on every arc of one path; trajectories are not IID samples.
    With ``closed_loop=False``, only the feedforward control is applied, subject
    to the same control limit. Both modes use identical samples for a given seed.
    """

    samples = options.monte_carlo_samples
    if samples < 2 or samples & (samples - 1):
        raise ValueError("Monte carlo samples must be a power of two and at least 2.")

    noise_dimension = Lc.shape[1]
    # Starting from one reproducible master seed, spawn three independent streams
    initial_seed, brownian_seed, pairing_seed = np.random.SeedSequence(
        options.monte_carlo_seed
    ).spawn(3)
    initial_state_sampler = qmc.Sobol(
        d=NX,
        scramble=True,
        seed=np.random.default_rng(initial_seed),
    )
    brownian_path_sampler = qmc.Sobol(
        d=options.n_arcs * 2 * noise_dimension,
        scramble=True,
        seed=np.random.default_rng(brownian_seed),
    )
    exponent = int(samples).bit_length() - 1
    initial_state_uniforms = initial_state_sampler.random_base2(m=exponent)
    brownian_path_uniforms = brownian_path_sampler.random_base2(m=exponent)
    # Keep the inverse normal CDF finite, including at a possible zero endpoint.
    eps = np.finfo(float).eps
    initial_state_normals = norm.ppf(np.clip(initial_state_uniforms, eps, 1.0 - eps)).T
    brownian_path_normals = norm.ppf(
        np.clip(brownian_path_uniforms, eps, 1.0 - eps)
    ).reshape(samples, options.n_arcs, 2 * noise_dimension)
    # Permute whole Brownian paths, preserving their internal Sobol coordinates.
    pairing = np.random.default_rng(pairing_seed).permutation(samples)
    brownian_path_normals = brownian_path_normals[pairing]

    L0 = np.linalg.cholesky(options.x0_covariance)
    states = solution.means[:, 0:1] + L0 @ initial_state_normals
    trajectories = np.empty((options.n_arcs + 1, NX, samples))
    trajectories[0] = states

    for k in range(options.n_arcs):
        control = np.broadcast_to(solution.feedforward[:, k : k + 1], (NU, samples))
        if closed_loop:
            deviation = states - solution.means[:, k : k + 1]
            control = control + solution.gains[k] @ deviation
        magnitude = np.linalg.norm(control, axis=0)
        saturated = magnitude > options.u_max
        scale = np.ones_like(magnitude)
        scale[saturated] = options.u_max / magnitude[saturated]
        applied_control = control * scale[None, :]
        arc_normals = brownian_path_normals[:, k, :].T
        increments = np.sqrt(options.dt) * arc_normals[:noise_dimension]
        space_time_levy_areas = (
            np.sqrt(options.dt / 12.0) * arc_normals[noise_dimension:]
        )
        states = np.asarray(
            _prescribed_sde_integration_step_batch(
                jnp.asarray(states),
                jnp.asarray(applied_control),
                jnp.asarray(increments),
                jnp.asarray(space_time_levy_areas),
                jnp.asarray(Lc),
                options.dt,
            )
        )
        trajectories[k + 1] = states

    state_deviations = trajectories - solution.means.T[:, :, None]
    # np.cov centers the sampled deviations and uses the N - 1 normalization.
    sampled_state_covariances = np.stack(
        [np.cov(deviations) for deviations in state_deviations]
    )
    return {
        "trajectories": trajectories,
        "state_covariances": sampled_state_covariances,
        "terminal_state_covariance": sampled_state_covariances[-1],
    }
