"""Differentiable RK9 maps with an exactly ballistic Mars phase."""
from types import SimpleNamespace
import jax.numpy as jnp
import diffrax
from xmds2_rk9 import XMDS2RK9
import stochastic_traj_opt_jax as common


def phase_case(mission, phase):
    scale, settings = mission.scales[phase], mission.settings
    force = scale.mass*scale.length/scale.time**2
    return SimpleNamespace(dynamics_model='heliocentric',
        test_case_id='psyche', time_unit=scale.time,
        length_unit=scale.length/1000, velocity_unit=scale.velocity/1000,
        max_thrust_nd=0. if mission.is_ballistic(phase) else settings.thrust_max_n/force,
        exhaust_velocity_nd=settings.isp_s*9.80665/scale.velocity,
        m0_wet=settings.wet_mass_kg, thrust_unit=force)


def nominal_map(case, substeps, eps, save_substeps=False):
    def rhs(t, x, args):
        command, dt = args
        magnitude = case.max_thrust_nd*jnp.sqrt(command@command+eps**2)
        return dt*common.vector_field(case, x, case.max_thrust_nd*command, magnitude)
    term, solver = diffrax.ODETerm(rhs), XMDS2RK9()
    def step(state, command, duration):
        result = diffrax.diffeqsolve(term, solver, t0=0., t1=1., dt0=1/substeps,
            y0=state, args=(command, duration), stepsize_controller=diffrax.ConstantStepSize(),
            saveat=(diffrax.SaveAt(ts=jnp.arange(1,substeps+1)/substeps)
                    if save_substeps else diffrax.SaveAt(t1=True)), max_steps=substeps+1,
            adjoint=common.step_adjoint())
        return result.ys if save_substeps else result.ys[0]
    return step


class Maps:
    def __init__(self, data, options, integration_refinement=1):
        self.data, self.options = data, options
        self.nominal, self.moments, self.cases, self.normalizations = [], [], [], []
        self.nominal_samples = []
        for i in range(3):
            case = phase_case(data.mission, i)
            norm = common.Normalization(data.covariance_scale/data.mission.scales[i].state,
                                        data.covariance_scale/data.mission.scales[i].state)
            self.cases.append(case)
            self.normalizations.append(norm)
            self.nominal.append(nominal_map(case, options.integrator_substeps, options.control_norm_eps))
            self.nominal_samples.append(nominal_map(case, options.integrator_substeps,
                                                    options.control_norm_eps, save_substeps=True))
            self.moments.append(common.build_propagation_arc_map(case, options, norm, integration_refinement))

    def deterministic(self, arc, mean, feed, days):
        """Nominal end state of one interval, in the next node's origin and scale."""
        duration = self.data.duration(arc, days)/self.cases[arc.phase].time_unit
        end = self.nominal[arc.phase](mean, feed, duration)
        return self.data.transform_end(arc, end, days)

    def deterministic_surface_samples(self, arc, mean, feed, days):
        """Clearance at the existing uniform RK9 endpoints, in the arc's origin."""
        mission = self.data.mission
        if arc.phase != 1:
            return jnp.zeros(0)
        radius = (mission.settings.mars_radius_m+500e3)/mission.scales[arc.phase].length
        duration = self.data.duration(arc, days)/self.cases[arc.phase].time_unit
        samples = self.nominal_samples[arc.phase](mean, feed, duration)
        return jnp.sum(samples[:,:3]**2,axis=1)-radius**2

    def stochastic(self, arc, mean, covariance, feed, gain, days):
        """Mean, covariance and executed-control covariance after one interval.

        Mars arcs propagate with zero command and gain, so their executed-control
        covariance is exactly zero.
        """
        duration = self.data.duration(arc, days)/self.cases[arc.phase].time_unit
        if arc.phase == 1:
            feed, gain = jnp.zeros(3), jnp.zeros((3,6))
        end, p, executed_cov = self.moments[arc.phase](mean, covariance, feed, gain, duration)
        if arc.phase == 1:
            executed_cov = jnp.zeros((3,3))
        return self.data.transform_end(arc, end, days), p, executed_cov
