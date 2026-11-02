"""Physical diagnostics and integration studies for a saved Psyche policy."""
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import stochastic_traj_opt_jax as common
from .psyche_maps import Maps
from .psyche_geometry import encounter_statistics


def nominal_linearization_accuracy(nlp, z):
    """Compare nominal map Jacobians and recomputed TVLQR gains at a fixed path.

    This is not a reoptimisation of the stochastic feedback gains. State,
    control and free-duration derivatives use the NLP's dimensionless units.
    """
    from .psyche_nlp import PsycheNLP
    from .psyche_run import weighted_tvlqr
    d, o = nlp.data, nlp.options
    variables = nlp.unpack(z)
    coarse = PsycheNLP(d, o)
    fine = PsycheNLP(d, replace(o, integrator_substeps=2*o.integrator_substeps,
                               sde_integrator_substeps=2*o.sde_integrator_substeps))
    nominal_z = coarse.x0.copy()
    for name in ('means', 'feedforward', 'times'):
        value = variables[name][d.powered_arcs] if name=='feedforward' else variables[name]
        nominal_z[coarse.slices[name]] = value.ravel()
    kc, _, jc = weighted_tvlqr(coarse, nominal_z)
    kf, _, jf = weighted_tvlqr(fine, nominal_z)
    phases = []
    for phase in range(3):
        ids = [a.control for a in d.arcs if a.phase == phase]
        errors = {name: [] for name in ('state', 'control', 'duration')}
        for index in ids:
            for name, cols in (('state', slice(0,7)), ('control', slice(7,10)),
                               ('duration', slice(-2,None))):
                delta = jf[index][:7,cols]-jc[index][:7,cols]
                errors[name].append(float(np.max(abs(delta))))
        change = kf[ids]-kc[ids]
        phases.append(dict(phase=phase, jacobian_max_absolute={k:max(v) for k,v in errors.items()},
            tvlqr_gain_max_absolute=float(np.max(abs(change))),
            tvlqr_gain_relative_frobenius=float(np.linalg.norm(change)/max(np.linalg.norm(kf[ids]),1e-30))))
    return dict(substeps=[o.integrator_substeps,2*o.integrator_substeps], phases=phases,
        qualification='Fixed nominal path; recomputed TVLQR gains, not reoptimised stochastic gains')


def recover(nlp, z):
    d = nlp.data
    v = nlp.unpack(z)
    cov = [d.initial_covariance]
    for i, packed in enumerate(v['factors']):
        factor = nlp.std[i+1, :, None]*np.asarray(common.LTRI.matrix(packed))
        cov.append(factor@factor.T)
    et = np.empty(d.n_nodes)
    et[0] = d.mission.t0
    weights = np.empty(d.n_intervals)
    controls = np.empty((d.n_intervals, 3))
    for a in d.arcs:
        duration = float(a.duration[0]+a.duration[1:]@v['times'])
        et[a.end] = et[a.start]+duration
        weights[a.end-1] = d.n_intervals*duration/(d.mission.tf-d.mission.t0)
        controls[a.end-1] = v['feedforward'][a.control]
    cov = np.array(cov)
    cov_si = cov*d.covariance_scale[None, :, None]*d.covariance_scale[None, None, :]
    return dict(**v, epochs_et=et, covariances=cov, covariance_si=cov_si,
                states_si=v['means']*d.node_scales, gains_normalized=v['gains']*nlp.gain_scale[:, None, None],
                interval_weights=weights, interval_controls=controls)


def physical_diagnostics(nlp, z):
    d, o = nlp.data, nlp.options
    x = recover(nlp, z)
    values = nlp.values(z)
    residual = values['mean_defects'].reshape(-1, 7)*d.node_scales[1:]
    ca = x['states_si'][d.ca_node]
    acc = -d.mission.scales[1].mu*ca[:3]/np.linalg.norm(ca[:3])**3
    encounter = {k:np.asarray(v) for k,v in encounter_statistics(jnp.asarray(ca[:6]),
        jnp.asarray(x['covariance_si'][d.ca_node,:6,:6]),jnp.asarray(acc),d.mission.scales[1].mu,
        d.mission.settings.mars_radius_m+500e3).items()}
    inv = d.terminal_inverse_std
    terminal = x['covariances'][-1,:6,:6]*np.outer(inv,inv)
    state_cost = 0.
    mean_control_cost = 0.
    for a in d.arcs:
        w = x['interval_weights'][a.end-1]
        state_cost += w*o.state_covariance_weight*np.trace(x['covariances'][a.start,:6,:6])/o.bryson_sigma_factor**2
        if a.phase != 1:
            feed = x['feedforward'][a.control]
            mean_control_cost += w*np.sqrt(np.dot(feed, feed)+o.control_norm_eps**2)
    # Standard cost: mean control + state covariance + weighted executed covariance.
    # Quantile cost: mean control + psi^-1 * executed-control spectral radius.
    total = float(values['objective'])
    terms = (dict(objective_state_covariance=float(state_cost),
                  objective_weighted_executed_control_covariance=total-mean_control_cost-state_cost,
                  objective_share_mean_control=mean_control_cost/total,objective_share_state_covariance=state_cost/total,
                  objective_share_control_covariance=(total-mean_control_cost-state_cost)/total)
             if o.objective == 'standard' else
             dict(objective_quantile_dispersion=float(values['objective']-mean_control_cost)))
    report = dict(objective=float(values['objective']), objective_kind=o.objective,
        objective_mean_control=float(mean_control_cost), **terms,
        mean_matching_position_m=float(np.max(np.linalg.norm(residual[:,:3],axis=1))),
        mean_matching_velocity_m_s=float(np.max(np.linalg.norm(residual[:,3:6],axis=1))),
        mean_matching_mass_kg=float(np.max(abs(residual[:,6]))),
        covariance_matching_scaled=float(np.max(abs(values['covariance_defects']))),
        terminal_psd_max_eigenvalue=float(np.linalg.eigvalsh(terminal)[-1]),
        terminal_std_si=np.sqrt(np.diag(x['covariance_si'][-1])).tolist(),
        mean_mass_min_kg=float(np.min(x['states_si'][:,6])),
        mean_final_mass_kg=float(x['states_si'][-1,6]),
        closest_approach_et=float(x['epochs_et'][d.ca_node]),
        mars_exit_et=float(x['epochs_et'][d.mars_exit_node]),
        closest_approach_mean_altitude_m=float(np.linalg.norm(ca[:3])-d.mission.settings.mars_radius_m),
        control_chance_covariance='executed; commanded dispersion and execution uncertainty included',
        maximum_control_chance_budget=float(np.max(values['control_chance'])),
        maximum_scaled_gain=float(np.max(abs(x['gains']))),
        encounter=encounter)
    report['nominal_command_energy_n2_s'] = float(sum(.5*d.mission.settings.thrust_max_n**2*
        (x['epochs_et'][i+1]-x['epochs_et'][i])*np.dot(u,u) for i,u in enumerate(x['interval_controls'])))
    report['mean_propellant_kg'] = float(x['states_si'][0,6]-x['states_si'][-1,6])
    jumps = []
    for node, entering in ((d.mars_entry_node,True),(d.mars_exit_node,False)):
        t = x['epochs_et'][node]
        planet, rate = jax.jvp(d.planet,(jnp.asarray(t),),(jnp.array(1.),))
        planet,rate = np.asarray(planet),np.asarray(rate)
        native = x['states_si'][node,:6]
        relative = native if entering else native-planet
        solar = relative+planet
        a_sun = -d.mission.scales[0].mu*solar[:3]/np.linalg.norm(solar[:3])**3
        a_mars = -d.mission.scales[1].mu*relative[:3]/np.linalg.norm(relative[:3])**3
        # f+ - DxR f- - R_t with identical held thrust. A separate commanded
        # update, if present, is already part of the scheduled control policy.
        jump = np.r_[rate[:3]-planet[3:6],a_mars-a_sun+rate[3:6]]*(1 if entering else -1)
        normal=relative[:3]/np.linalg.norm(relative[:3])
        radial_speed=float(normal@(relative[3:6]+(planet[3:6]-rate[:3] if entering else 0.)))
        time_gradient=-normal/radial_speed
        time_sigma=float(np.sqrt(max(time_gradient@x['covariance_si'][node,:3,:3]@time_gradient,0.)))
        jumps.append(dict(event='entry' if entering else 'exit',velocity_jump_m_s=jump[:3],
                          acceleration_jump_m_s2=jump[3:],
                          acceleration_jump_norm_m_s2=float(np.linalg.norm(jump[3:])),
                          radial_crossing_speed_m_s=radial_speed,
                          linearized_crossing_time_std_s=time_sigma,
                          velocity_effect_scale_m_s=float(np.linalg.norm(jump[3:])*time_sigma),
                          qualification='Local first-order scale; no propagation of a saltation correction'))
    report['gravity_frame_vector_field_mismatch'] = jumps
    report['control_chance_budget_by_interval'] = [
        (0. if d.control_indices[a.control]<0 else float(values['control_chance'][d.control_indices[a.control]]))
        for a in d.arcs]
    # Chance budget minus the smoothed feedforward norm: psi^-1 times the
    # executed-control spectral radius, the predicted applied-thrust deviation.
    report['control_dispersion_bound_by_interval'] = [
        (0. if d.control_indices[a.control]<0 else
         budget-float(np.sqrt(np.dot(x['feedforward'][a.control],x['feedforward'][a.control])+o.control_norm_eps**2)))
        for a,budget in zip(d.arcs,report['control_chance_budget_by_interval'])]
    return x, report


def moment_accuracy(nlp, z, *, change_noise_grid=False):
    """Compare the NLP's RK9 moment map with one using twice as many steps.

    With change_noise_grid=False the UT points, weights and Wiener paths are
    identical: each Wiener increment is divided equally between two RK9 steps.
    True selects twice as many independent Wiener increments and hence a
    different augmented UT ensemble; its discrepancy is not pure ODE error.
    The closest-approach clearance change is evaluated on the arc ending at CA.
    """
    d, o = nlp.data, nlp.options
    x = recover(nlp, z)
    fine_options = replace(o,integrator_substeps=2*o.integrator_substeps,
        sde_integrator_substeps=2*o.sde_integrator_substeps) if change_noise_grid else o
    fine = Maps(d,fine_options,integration_refinement=1 if change_noise_grid else 2)
    rho = d.mission.settings.mars_radius_m+500e3
    mu = d.mission.scales[1].mu

    def radius(executed):
        return common.psi_inverse(3,o.violation_parameter)*common.spectral_radius(
            executed,o.spectral_radius_floor,o.eigenvalue_smoothing)

    def clearance(end, covariance):
        state = end*jnp.asarray(d.node_scales[d.ca_node])
        scale = jnp.asarray(d.covariance_scale[:6])
        acc = -mu*state[:3]/jnp.linalg.norm(state[:3])**3
        return encounter_statistics(state[:6],covariance[:6,:6]*scale[:,None]*scale[None,:],acc,mu,rho)[
            'chance_clearance']

    functions = {}
    statistics = [[] for _ in range(3)]
    for a in d.arcs:
        key = (a.phase,a.switch,a.end == d.ca_node)
        if key not in functions:
            def compare(mean,p,u,k,t,duration,template=a):
                arc = replace(template,duration=duration)
                coarse = nlp.maps.stochastic(arc,mean,p,u,k,t)
                refined = fine.stochastic(arc,mean,p,u,k,t)
                encounter_change = (clearance(*refined[:2])-clearance(*coarse[:2])
                                    if template.end == d.ca_node else jnp.array(0.))
                return (*(b-c for c,b in zip(coarse,refined)),radius(refined[2])-radius(coarse[2]),
                        encounter_change)
            functions[key] = jax.jit(compare)
        dy,dp,dce,dchance,dencounter = map(np.asarray,functions[key](x['means'][a.start],x['covariances'][a.start],
            x['feedforward'][a.control],x['gains_normalized'][a.control],x['times'],a.duration))
        dy_si = dy*d.node_scales[a.end]
        dp_si = dp*d.covariance_scale[:,None]*d.covariance_scale[None,:]
        std = nlp.std[a.end]
        statistics[a.phase].append(dict(position_m=float(np.linalg.norm(dy_si[:3])),
            velocity_m_s=float(np.linalg.norm(dy_si[3:6])),mass_kg=float(abs(dy_si[6])),
            covariance_normalized=float(np.max(abs(dp))),
            covariance_seed_scaled=float(np.max(abs(dp/std[:,None]/std[None,:]))),
            position_variance_m2=float(np.max(abs(dp_si[:3,:3]))),
            velocity_variance_m2_s2=float(np.max(abs(dp_si[3:6,3:6]))),
            executed_covariance_normalized=float(np.max(abs(dce))),
            executed_chance_budget_difference=float(abs(dchance)),
            executed_chance_budget_difference_n=float(abs(dchance)*d.mission.settings.thrust_max_n),
            ca_clearance_difference_normalized=float(abs(dencounter)),
            ca_clearance_difference_m=float(abs(dencounter)*rho)))
    return dict(substeps=[o.sde_integrator_substeps,2*o.sde_integrator_substeps],
        comparison='independent finer Wiener/UT grid' if change_noise_grid else 'identical UT and prescribed Wiener paths',
        phases=[dict(phase=i,**{k:max(row[k] for row in rows) for k in rows[0]}) for i,rows in enumerate(statistics)])
