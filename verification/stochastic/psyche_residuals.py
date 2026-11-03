"""Independent residual evaluation for the three physical Psyche phases.

Mars has no thrust or gain decision variables. Closest approach is an ordinary
state/covariance shooting node; moments remain continuous without a reset.
Every arc maps its start node to the next node; arcs with the same phase and
origin switch share one vectorised JAX function.

The energy reoptimisation keeps the two Mars durations (entry -> CA -> exit) as
variables with the CA, SOI-exit, outward-crossing and duration rows. The
stochastic NLP fixes them (`mars_days`, normally the reoptimised values): the
ballistic mean passage is set by the fixed Mars entry state. q > 0 and C3 > 0
are then checked after solving. `mars_days=None` keeps them free.
"""

from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import stochastic_trajectory_optimization.dynamics as common
from verification.stochastic.psyche_model import encounter_statistics
from verification.stochastic.psyche_model import Maps


def mars_event_rows(data, ca, sun_exit, days):
    """SOI exit radius, CA radial velocity, outward exit, Mars duration, q and C3.

    `ca` and `sun_exit` are SI states (Mars- and Sun-centred). q and C3 must be
    strictly positive for a hyperbolic local minimum at CA.
    """
    mission = data.mission
    exit_state = sun_exit[:6] - data.planet(mission.tm + 86400 * jnp.sum(days))
    r, vel = ca[:3], ca[3:6]
    # Ballistic Mars: the closest-approach acceleration is Mars gravity alone.
    acc = -mission.scales[1].mu * r / jnp.linalg.norm(r) ** 3
    q = vel @ vel + r @ acc
    c = vel @ vel - 2 * mission.scales[1].mu / jnp.linalg.norm(r)
    return jnp.array(
        [
            (jnp.linalg.norm(exit_state[:3]) / mission.soi[499]) - 1,
            (r @ vel) / (mission.scales[1].length * mission.scales[1].velocity),
            (exit_state[:3] @ exit_state[3:6])
            / (mission.soi[499] * mission.scales[1].velocity),
            jnp.sum(days),
            q / 1e6,
            c / 1e6,
        ]
    )


class PsycheResiduals:
    def __init__(
        self,
        data,
        options,
        seed=None,
        covariance_std=None,
        mars_days=None,
        terminal_covariance_cost=False,
    ):
        self.data, self.options = data, options
        self.terminal_covariance_cost = terminal_covariance_cost
        self.maps = Maps(data, options)
        self.stochastic = seed is not None
        if mars_days is not None and not self.stochastic:
            raise ValueError("The energy reoptimisation keeps the Mars durations free")
        self.mars_days = (
            None if mars_days is None else np.asarray(mars_days, dtype=float)
        )
        self.free_days = self.mars_days is None
        self.node_count, self.control_count = data.n_nodes, data.n_controls
        self.surface_nodes = np.flatnonzero(np.asarray(data.node_origins) == 1)
        self.surface_arcs = [k for k, arc in enumerate(data.arcs) if arc.phase == 1]
        self.surface_slots = {k: slot for slot, k in enumerate(self.surface_arcs)}
        self.surface_radius_squared = (
            (data.mission.settings.mars_radius_m + 500e3)
            / data.node_scales[self.surface_nodes, 0]
        ) ** 2
        n, c = self.node_count, data.n_powered_controls
        self.slices, offset = {}, 0
        sizes = dict(means=7 * n, feedforward=3 * c)
        if self.free_days:
            sizes["times"] = 2
        if self.stochastic:
            sizes.update(gains=18 * c, factors=28 * (n - 1), margin=21)
        for key, count in sizes.items():
            self.slices[key] = slice(offset, offset + count)
            offset += count
        self.size = offset
        self.std = np.ones((n, 7)) if covariance_std is None else covariance_std
        self.gain_scale = (
            np.ones(self.control_count) if seed is None else seed["gain_scale"]
        )
        self.x0 = np.zeros(offset)
        self.x0[self.slices["means"]] = (
            data.seed_states if seed is None else seed["means"]
        ).ravel()
        self.x0[self.slices["feedforward"]] = (
            data.seed_controls if seed is None else seed["feedforward"]
        )[data.powered_arcs].ravel()
        if self.free_days:
            self.x0[self.slices["times"]] = (
                data.seed_durations if seed is None else seed["times"]
            )
        if self.stochastic:
            self.x0[self.slices["gains"]] = (
                seed["gains"] / self.gain_scale[:, None, None]
            )[data.powered_arcs].ravel()
            factors = []
            for i in range(1, n):
                scaled = seed["covariances"][i] / np.outer(self.std[i], self.std[i])
                factor = np.linalg.cholesky((scaled + scaled.T) / 2 + 1e-14 * np.eye(7))
                factors.append(np.asarray(common.LTRI.pack(factor)))
            self.x0[self.slices["factors"]] = np.array(factors).ravel()
            slack = np.eye(6) - seed["covariances"][-1, :6, :6] * np.outer(
                data.terminal_inverse_std, data.terminal_inverse_std
            )
            val, vec = np.linalg.eigh((slack + slack.T) / 2)
            margin = np.linalg.cholesky(
                (vec * np.maximum(val, 0)) @ vec.T
                + options.terminal_margin_floor**2 * np.eye(6)
            )
            self.x0[self.slices["margin"]] = np.asarray(common.MTRI.pack(margin))
        self.local_columns = []
        self.groups = {}
        for index, arc in enumerate(data.arcs):
            cols = np.r_[
                self.indices("means", arc.start, 7),
                self.indices("feedforward", index, 3),
            ]
            if self.stochastic:
                lcols = (
                    self.indices("factors", arc.start - 1, 28)
                    if arc.start
                    else np.full(28, -1)
                )
                cols = np.r_[cols, self.indices("gains", index, 18), lcols]
            cols = np.r_[cols, self.day_columns()]
            self.local_columns.append(cols)
            self.groups.setdefault((arc.phase, arc.switch), []).append(index)
        self.local_columns = np.array(self.local_columns, dtype=int)
        # Constant inputs where a column is not a variable: Mars controls and the
        # fixed initial factor are zero (see make_local), fixed days are values.
        self.local_fill = np.zeros(self.local_columns.shape)
        if not self.free_days:
            self.local_fill[:, -2:] = self.mars_days
        self.functions = {}
        for key, indices in self.groups.items():
            template = data.arcs[indices[0]]
            function = self.make_local(template)
            self.functions[key] = (
                jax.jit(jax.vmap(function)),
                jax.jit(jax.vmap(jax.jacrev(function, argnums=0))),
            )
        # Global inputs: CA mean, then (free days) the Mars-exit mean and the
        # days, then (stochastic) the CA and final covariance factors and margin.
        self.global_columns = self.indices("means", data.ca_node, 7)
        if self.free_days:
            self.global_columns = np.r_[
                self.global_columns,
                self.indices("means", data.mars_exit_node, 7),
                self.day_columns(),
            ]
        if self.stochastic:
            self.global_columns = np.r_[
                self.global_columns,
                self.indices("factors", data.ca_node - 1, 28),
                self.indices("factors", n - 2, 28),
                np.arange(self.slices["margin"].start, self.slices["margin"].stop),
            ]
        self.global_value = jax.jit(self.global_output)
        self.global_derivative = jax.jit(jax.jacrev(self.global_output))
        event_lower, event_upper = (
            ([0.0, 0.0, 0.0, 0.05, 0.0, 0.0], [0.0, 0.0, np.inf, 10.0, np.inf, np.inf])
            if self.free_days
            else ([], [])
        )
        self.global_lower = np.array(
            event_lower + ([0.0] + [0.0] * 21 if self.stochastic else [])
        )
        self.global_upper = np.array(
            event_upper + ([np.inf] + [0.0] * 21 if self.stochastic else [])
        )
        self._last = None
        # Fixed sparse topology: even numerically zero local entries are retained.

    def day_columns(self):
        return (
            np.arange(self.slices["times"].start, self.slices["times"].stop)
            if self.free_days
            else np.full(2, -1)
        )

    def indices(self, group, block, width):
        if group in ("feedforward", "gains"):
            block = self.data.control_indices[block]
            if block < 0:
                return np.full(width, -1, dtype=int)
        return self.slices[group].start + block * width + np.arange(width)

    def unpack(self, z):
        result = {name: z[sl] for name, sl in self.slices.items()}
        result["means"] = result["means"].reshape(self.node_count, 7)
        feed = np.zeros((self.control_count, 3))
        feed[self.data.powered_arcs] = result["feedforward"].reshape(-1, 3)
        result["feedforward"] = feed
        if self.stochastic:
            gains = np.zeros((self.control_count, 3, 6))
            gains[self.data.powered_arcs] = result["gains"].reshape(-1, 3, 6)
            result["gains"] = gains
            result["factors"] = result["factors"].reshape(self.node_count - 1, 28)
        if not self.free_days:
            result["times"] = self.mars_days.copy()
        return result

    def local_inputs(self, z):
        # Negative indices are constants: zero Mars controls, the fixed initial
        # covariance (selected by a parameter flag) and fixed Mars days.
        return np.where(
            self.local_columns >= 0,
            np.asarray(z)[np.maximum(self.local_columns, 0)],
            self.local_fill,
        )

    def global_inputs(self, z):
        return np.where(
            self.global_columns >= 0,
            np.asarray(z)[np.maximum(self.global_columns, 0)],
            0.0,
        )

    def mars_events(self, z):
        """Event rows of the mean trajectory (see mars_event_rows), in any mode."""
        means = self.unpack(np.asarray(z))
        mission = self.data.mission
        return np.asarray(
            mars_event_rows(
                self.data,
                jnp.asarray(
                    means["means"][self.data.ca_node] * mission.scales[1].state
                ),
                jnp.asarray(
                    means["means"][self.data.mars_exit_node] * mission.scales[2].state
                ),
                jnp.asarray(means["times"]),
            )
        )

    def parameters(self, indices):
        """Per-arc constants: duration coefficients, start/end node standard
        deviations, gain scale and an initial-covariance flag."""
        rows = []
        for index in indices:
            arc = self.data.arcs[index]
            rows.append(
                np.r_[
                    arc.duration,
                    self.std[arc.start],
                    self.std[arc.end],
                    self.gain_scale[index],
                    float(arc.start == 0),
                ]
            )
        return jnp.asarray(np.array(rows))

    def make_local(self, template):
        """Local outputs: next mean (7), then either the next packed covariance
        (28, stochastic) or Mars altitude samples (energy), then the
        control-norm/chance value and the cost contribution."""
        data, options = self.data, self.options

        def evaluate(v, p):
            arc = replace(template, duration=p[:3])
            mean, feed, days = v[:7], v[7:10], v[-2:]
            norm = (
                0.0
                if arc.phase == 1
                else jnp.sqrt(feed @ feed + options.control_norm_eps**2)
            )
            dt = data.duration(arc, days)
            if not self.stochastic:
                end = self.maps.deterministic(arc, mean, feed, days)
                surfaces = self.maps.deterministic_surface_samples(
                    arc, mean, feed, days
                )
                cost = 0.5 * dt / (data.mission.tf - data.mission.t0) * (feed @ feed)
                return jnp.concatenate((end, surfaces, jnp.array([norm, cost])))
            gain = v[10:28].reshape(3, 6) * p[17]
            root = p[3:10, None] * common.LTRI.matrix(v[28:56])
            covariance = jnp.where(
                p[18] > 0.5, jnp.asarray(data.initial_covariance), root @ root.T
            )
            end, end_covariance, executed = self.maps.stochastic(
                arc, mean, covariance, feed, gain, days
            )
            end_std = p[10:17]
            packed = common.LTRI.pack(
                jnp.linalg.cholesky(
                    end_covariance / end_std[:, None] / end_std[None, :]
                )
            )
            chance = (
                0.0
                if arc.phase == 1
                else norm
                + common.psi_inverse(3, options.violation_parameter)
                * common.spectral_radius(
                    executed,
                    options.spectral_radius_floor,
                    options.eigenvalue_smoothing,
                )
            )
            running = norm + options.control_covariance_weight * jnp.trace(executed)
            # Duration-weighted cost: N_intervals*dt/T_total times the running cost
            # (objective "quantile": the thrust-chance quantile; zero on Mars arcs).
            cost = dt * (
                chance
                if options.objective == "quantile"
                else running
                + options.state_covariance_weight
                * jnp.trace(covariance[:6, :6])
                / options.bryson_sigma_factor**2
            )
            cost *= data.n_intervals / (data.mission.tf - data.mission.t0)
            return jnp.concatenate((end, packed, jnp.array([chance, cost])))

        return evaluate

    def global_output(self, v):
        data, mission = self.data, self.data.mission
        ca = v[:7] * jnp.asarray(mission.scales[1].state)
        values, k = [], 7
        if self.free_days:
            # q and C3 must be strictly positive in the accepted result. Nonnegative
            # NLP bounds are supplemented by a domain validity check after solving.
            sun_exit = v[7:14] * jnp.asarray(mission.scales[2].state)
            values.append(mars_event_rows(data, ca, sun_exit, v[14:16]))
            k = 16
        if self.stochastic:
            r = ca[:3]
            acc = -mission.scales[1].mu * r / jnp.linalg.norm(r) ** 3
            fca = jnp.asarray(self.std[data.ca_node])[:, None] * common.LTRI.matrix(
                v[k : k + 28]
            )
            pca = fca @ fca.T
            d = jnp.asarray(data.covariance_scale[:6])
            encounter = encounter_statistics(
                ca[:6],
                pca[:6, :6] * d[:, None] * d[None, :],
                acc,
                mission.scales[1].mu,
                mission.settings.mars_radius_m + 500e3,
            )
            ff = jnp.asarray(self.std[-1])[:, None] * common.LTRI.matrix(
                v[k + 28 : k + 56]
            )
            pfinal = ff @ ff.T
            inv = jnp.asarray(data.terminal_inverse_std)
            margin = common.MTRI.matrix(v[k + 56 : k + 77])
            terminal = (
                jnp.eye(6)
                - pfinal[:6, :6] * inv[:, None] * inv[None, :]
                - margin @ margin.T
            )
            values += [
                encounter["chance_clearance"][None],
                jnp.zeros(21)
                if self.terminal_covariance_cost
                else common.MTRI.pack(terminal),
            ]
        return jnp.concatenate(values)

    def evaluate_local(self, z, derivatives=False):
        local = self.local_inputs(z)
        results = [None] * self.control_count
        for key, indices in self.groups.items():
            function = self.functions[key][int(derivatives)]
            out = np.asarray(
                function(jnp.asarray(local[indices]), self.parameters(indices))
            )
            for i, value in zip(indices, out):
                results[i] = value
        return results

    def values(self, z):
        variables = self.unpack(np.asarray(z))
        local = self.evaluate_local(z)
        mean = np.zeros((self.node_count - 1, 7))
        covariance = np.zeros((self.node_count - 1, 28))
        chances, surfaces, cost = [], [], 0.0
        for index, value in enumerate(local):
            end = self.data.arcs[index].end
            mean[end - 1] = variables["means"][end] - value[:7]
            if self.stochastic:
                covariance[end - 1] = variables["factors"][end - 1] - value[7:35]
            if self.data.control_indices[index] >= 0:
                chances.append(value[-2])
            if not self.stochastic:
                surfaces.extend(value[7:-2])
            cost += value[-1]
        if self.stochastic and self.terminal_covariance_cost:
            factor = self.std[-1, :, None] * np.asarray(
                common.LTRI.matrix(variables["factors"][-1])
            )
            target = (factor @ factor.T)[:6, :6] * np.outer(
                self.data.terminal_inverse_std, self.data.terminal_inverse_std
            )
            cost += np.trace(target) / self.options.bryson_sigma_factor**2
        result = dict(
            objective=cost,
            mean_defects=mean.ravel(),
            control_chance=np.asarray(chances),
            events=np.asarray(self.global_value(jnp.asarray(self.global_inputs(z)))),
        )
        if self.stochastic:
            result["covariance_defects"] = covariance.ravel()
        else:
            node_surfaces = (
                np.sum(variables["means"][self.surface_nodes, :3] ** 2, axis=1)
                - self.surface_radius_squared
            )
            result["surface_clearance"] = np.r_[surfaces, node_surfaces]
        return result
