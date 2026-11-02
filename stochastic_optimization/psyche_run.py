"""Psyche energy restoration -> TVLQR -> nonlinear stochastic shooting."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import stochastic_traj_opt_jax as common
from .psyche_inputs import PsycheInput, ROOT, MESH_REFINEMENT
from .psyche_maps import Maps
from .psyche_nlp import PsycheNLP


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')


def weighted_tvlqr(nlp, z):
    """Duration-weighted Bryson TVLQR on the nominal shooting Jacobians.

    Running weights are w_k*Q and w_k*R with w_k = N*dt_k/T_total. Ballistic Mars
    arcs keep K = 0 but still propagate the Riccati matrix, so the preceding
    cruise feedback accounts for the uncontrolled flyby.
    """
    data, options = nlp.data, nlp.options
    days = nlp.unpack(z)['times']
    jacobians = nlp.evaluate_local(z, True)
    total = data.mission.tf-data.mission.t0
    qbase = options.state_covariance_weight*np.eye(6)/options.bryson_sigma_factor**2
    rbase = options.control_covariance_weight*np.eye(3)
    s = np.diag(data.terminal_inverse_std**2)/options.bryson_sigma_factor**2
    gains, gain_scale = np.empty((data.n_controls,3,6)), np.empty(data.n_controls)
    for index in range(data.n_controls-1,-1,-1):
        arc, jac = data.arcs[index], jacobians[index]
        input_scale = data.covariance_scale/data.node_scales[arc.start]
        output_scale = data.covariance_scale/data.node_scales[arc.end]
        a = jac[:6,:6]*input_scale[None,:6]/output_scale[:6,None]
        b = jac[:6,7:10]/output_scale[:6,None]
        gain_scale[index] = 1. if arc.phase == 1 else 1/max(np.linalg.norm(b,2),1e-12)
        w = data.n_intervals*float(arc.duration[0]+arc.duration[1:]@days)/total
        q, r = w*qbase, w*rbase
        k = np.zeros((3,6)) if arc.phase == 1 else -np.linalg.solve(r+b.T@s@b,b.T@s@a)
        gains[index] = k
        closed = a+b@k
        s = q+k.T@r@k+closed.T@s@closed
        s = (s+s.T)/2
    return gains, gain_scale, jacobians


def make_seed(nlp, z):
    data = nlp.data
    variables = nlp.unpack(z)
    means, feeds, days = variables['means'], variables['feedforward'], variables['times']
    gains, gain_scale, _ = weighted_tvlqr(nlp, z)
    # Sequential propagation carries covariance across every origin translation.
    covariances = np.empty((data.n_nodes,7,7))
    covariances[0] = data.initial_covariance
    propagated = means.copy()
    moment_functions = {}
    for arc in data.arcs:
        key = (arc.phase,arc.switch)
        if key not in moment_functions:
            def f(mean,p,u,k,days,duration,template=arc):
                return nlp.maps.stochastic(replace(template,duration=duration),mean,p,u,k,days)[:2]
            moment_functions[key] = jax.jit(f)
        end, covariance = moment_functions[key](jnp.asarray(means[arc.start]),jnp.asarray(covariances[arc.start]),
            jnp.asarray(feeds[arc.control]),jnp.asarray(gains[arc.control]),jnp.asarray(days),jnp.asarray(arc.duration))
        propagated[arc.end] = np.asarray(end)
        covariances[arc.end] = np.asarray(covariance)
    if not np.all(np.isfinite(covariances)):
        raise FloatingPointError('Nonfinite TVLQR covariance seed')
    return dict(means=means,feedforward=feeds,times=days,gains=gains,gain_scale=gain_scale,
                covariances=covariances,propagated_means=propagated)


def integration_report(nlp, z):
    data, options = nlp.data, nlp.options
    finer = Maps(data, replace(options,integrator_substeps=2*options.integrator_substeps,
                                sde_integrator_substeps=2*options.sde_integrator_substeps))
    variables = nlp.unpack(z)
    by_phase = [[] for _ in range(3)]
    functions = {}
    for arc in data.arcs:
        key = (arc.phase,arc.switch)
        if key not in functions:
            def compare(x,u,t,duration,template=arc):
                a=replace(template,duration=duration)
                return finer.deterministic(a,x,u,t)-nlp.maps.deterministic(a,x,u,t)
            functions[key]=jax.jit(compare)
        error=np.asarray(functions[key](variables['means'][arc.start],variables['feedforward'][arc.control],
                                        variables['times'],arc.duration))
        by_phase[arc.phase].append(error*data.node_scales[arc.end])
    return [dict(phase=i,position_m=float(np.max(np.linalg.norm(np.array(e)[:,:3],axis=1))),
                 velocity_m_s=float(np.max(np.linalg.norm(np.array(e)[:,3:6],axis=1))),
                 mass_kg=float(np.max(abs(np.array(e)[:,6])))) for i,e in enumerate(by_phase)]


def postprocess(output, samples=50000):
    """Resume diagnostics without another solve; never label a candidate accepted."""
    from .psyche_diagnostics import physical_diagnostics, moment_accuracy, nominal_linearization_accuracy
    from .psyche_plots import plot_diagnostics
    from .psyche_monte_carlo import run_monte_carlo, MONTE_CARLO_MIN_SUBSTEPS
    output=Path(output)
    config=json.loads((output/'configuration.json').read_text())
    data=PsycheInput(config['deterministic_reference'],config.get('mesh_refinement',1))
    if hashlib.sha256(data.path.read_bytes()).hexdigest()!=config['deterministic_reference_sha256']:
        raise ValueError('The deterministic reference changed since the stochastic solve')
    options=common.Options(**config['options'])
    with np.load(output/'tvlqr_seed.npz') as f:
        seed={k:f[k] for k in f.files}
    with np.load(output/'solution.npz') as f:
        z,std,days=f['z'],f['covariance_std'],f['times']
    nlp=PsycheNLP(data,options,seed=seed,covariance_std=std,
                  mars_days=None if config.get('free_mars_events',False) else days)
    arrays,physical=physical_diagnostics(nlp,z)
    result=json.loads((output/'result.json').read_text())
    physical['converged']=bool(result['converged'])
    np.savez_compressed(output/'physical_solution.npz',**arrays)
    write(output/'physical_diagnostics.json',physical)
    table=np.column_stack((arrays['epochs_et'],np.array(data.node_origins),arrays['states_si'],
                           np.sqrt(np.diagonal(arrays['covariance_si'],axis1=1,axis2=2))))
    np.savetxt(output/'psyche.csv',table,delimiter=',',comments='',
        header='et_tdb_s,origin_phase,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,m_kg,sx_m,sy_m,sz_m,svx_m_s,svy_m_s,svz_m_s,sm_kg')
    plot_diagnostics(data,arrays,physical,output)
    if not result['converged']:
        print('Candidate diagnostics exported; stochastic NLP was not accepted.',flush=True)
        return physical
    write(output/'moment_accuracy_fixed_paths.json',moment_accuracy(nlp,z))
    write(output/'moment_accuracy_finer_noise.json',moment_accuracy(nlp,z,change_noise_grid=True))
    write(output/'nominal_linearization_accuracy.json',nominal_linearization_accuracy(nlp,z))
    substeps=max(MONTE_CARLO_MIN_SUBSTEPS,options.monte_carlo_sde_integrator_substeps or options.sde_integrator_substeps)
    ensembles=[]
    for seed_number in (42,43,44):
        t=time.perf_counter()
        mc,stats=run_monte_carlo(data,arrays,nlp.gain_scale,samples=samples,seed=seed_number,substeps=substeps)
        stats['wall_seconds']=time.perf_counter()-t
        np.savez_compressed(output/f'monte_carlo_{seed_number}.npz',**mc)
        write(output/f'monte_carlo_{seed_number}.json',stats)
        ensembles.append(mc)
        print(f'Psyche Monte Carlo seed {seed_number}: {stats["unsafe_altitude_count"]}/{samples} unsafe, '
              f'{stats["invalid_hyperbolic_count"]} invalid geometries',flush=True)
    fine,stats=run_monte_carlo(data,arrays,nlp.gain_scale,samples=samples,seed=42,substeps=2*substeps)
    np.savez_compressed(output/'monte_carlo_42_finer.npz',**fine)
    difference=dict(position_m=float(np.max(np.linalg.norm(fine['terminal_si'][:,:3]-ensembles[0]['terminal_si'][:,:3],axis=1))),
        velocity_m_s=float(np.max(np.linalg.norm(fine['terminal_si'][:,3:6]-ensembles[0]['terminal_si'][:,3:6],axis=1))),
        altitude_m=float(np.max(abs(fine['minimum_altitude_m']-ensembles[0]['minimum_altitude_m']))),
        ca_time_s=float(np.max(abs(fine['closest_et']-ensembles[0]['closest_et']))))
    write(output/'monte_carlo_42_finer.json',dict(**stats,coupled_differences=difference))
    plot_diagnostics(data,arrays,physical,output,ensembles)
    return physical


def run(output=None, reference=None, substeps=1, stage='all', samples=50000,
        solver_log_directory=None, eigenvalue_smoothing=None, minor_print_level=0,
        reduced_hessian_dimension=None, nominal_feasibility_tolerance=None, free_mars_events=False,
        objective='standard', mesh_refinement=MESH_REFINEMENT):
    if stage=='diagnostics':
        if output is None:
            raise ValueError('Diagnostics require the saved --output directory')
        return postprocess(output,samples)
    if not isinstance(substeps,(int,np.integer)) or substeps < 1:
        raise ValueError('Psyche RK9 substeps must be a positive integer')
    data = PsycheInput(reference, mesh_refinement)
    output = Path(output or ROOT/'output/psyche_mission_stochastic_traj_opt_jax'/jax.default_backend())
    output.mkdir(parents=True,exist_ok=True)
    # Earth-Mars problem defaults; Psyche's own tracking weights (not tuned yet: w_P = 1, w = 1000).
    options = replace(common.default_options('earth_mars'),initial_position_std_km=2000.,objective=objective,
        state_covariance_weight=1.,control_covariance_weight=1000.,
        integrator_substeps=substeps,sde_integrator_substeps=substeps,
        monte_carlo_sde_integrator_substeps=substeps,
        monte_carlo_samples=samples,solver_log_directory=str(solver_log_directory or output/'solver_logs'),
        minor_print_level=minor_print_level)
    if eigenvalue_smoothing is not None:
        options=replace(options,eigenvalue_smoothing=eigenvalue_smoothing)
    snopt_overrides={}
    if reduced_hessian_dimension is not None:
        if reduced_hessian_dimension<1:
            raise ValueError('Reduced Hessian dimension must be positive')
        snopt_overrides={'Reduced Hessian dimension':int(reduced_hessian_dimension),
            'Minor print level':1,'Print frequency':1000,'Summary frequency':1000}
    config=dict(options=asdict(options),deterministic_reference=str(data.path),mesh_refinement=data.mesh_refinement,
        deterministic_reference_sha256=hashlib.sha256(data.path.read_bytes()).hexdigest(),
        initial_mesh=[len(p['mesh'])-1 for p in data.solution['phases']],
        numerical_intervals=data.n_intervals,control_arcs=data.n_powered_controls,
        fixed_events=dict(Earth_SOI_exit=data.mission.t0,Mars_entry=data.mission.tm,arrival=data.mission.tf),
        closest_approach_node=data.ca_node,ballistic_mars=True,
        launch_covariance_si=data.initial_covariance_si, covariance_scale_si=data.covariance_scale,
        minimum_mars_altitude_m=500e3,flyby_risk=.001,terminal_position_std_m=2e6,terminal_velocity_std_m_s=2.,
        bplane_eigenvalue_smoothing=0.,
        initial_event_epochs_et=dict(
            closest_approach=data.mission.tm+86400*float(data.seed_durations[0]),
            Mars_SOI_exit=data.mission.tm+86400*float(np.sum(data.seed_durations))),
        event_initialization='Deterministic energy closest approach and Mars exit; free during energy reoptimisation',
        free_mars_events=bool(free_mars_events),
        mars_events=('free variables in the stochastic solve' if free_mars_events else
                     'fixed in the stochastic solve at the energy reoptimisation epochs'),
        stochastic_cost='N_intervals/total_duration * sum duration * (q_eta + trace(P6)/9 + trace(C_executed))',
        chance_control_covariance='executed; commanded dispersion and execution uncertainty included',
        surfaces='no mean surface constraints; mean mass floor retained',
        nominal_surfaces='Mars altitude >=500 km at mesh nodes and uniform RK9 endpoints; no Sun exclusion',
        passive_loss_of_thrust='not implemented; requires explicit later request',
        stochastic_switches='scheduled nominal epochs; Monte Carlo sample-specific crossings are a different hybrid model',
        device=str(jax.devices()),snopt_numerical_overrides=snopt_overrides,
        gain_initialization='Duration-weighted Bryson TVLQR; ballistic Mars has K=0; no gain projection',
        nominal_feasibility_tolerance=nominal_feasibility_tolerance)
    write(output/'configuration.json',config)
    nominal=PsycheNLP(data,options)
    nominal_overrides = {} if nominal_feasibility_tolerance is None else {
        'Major feasibility tolerance': nominal_feasibility_tolerance,
        'Minor feasibility tolerance': nominal_feasibility_tolerance}
    if nominal_feasibility_tolerance is not None and not 0 < nominal_feasibility_tolerance < 1:
        raise ValueError('Nominal feasibility tolerance must be between zero and one')
    z,report=nominal.solve('psyche_reference',solver_overrides=nominal_overrides)
    reference=nominal.unpack(z)
    feeds=reference['feedforward']
    energy=sum(.5*data.mission.settings.thrust_max_n**2*float(data.duration(a,reference['times']))
               *np.dot(feeds[a.control],feeds[a.control]) for a in data.arcs)
    report.update(energy_n2_s=float(energy),
                  fuel_kg=float(data.mission.settings.wet_mass_kg*(reference['means'][0,6]-reference['means'][-1,6])),
                  mars_days=reference['times'].tolist())
    np.savez_compressed(output/'reference.npz',z=z,**reference)
    write(output/'reference_result.json',report)
    print(json.dumps(report),flush=True)
    if not report['converged']:
        return dict(converged=False,stage='energy_restoration',report=report)
    write(output/'integration_accuracy.json',dict(substeps=[substeps,2*substeps],local=integration_report(nominal,z)))
    if stage=='nominal':
        return report
    seed=make_seed(nominal,z)
    np.savez_compressed(output/'tvlqr_seed.npz',**seed)
    std=np.sqrt(np.maximum(np.diagonal(seed['covariances'],axis1=1,axis2=2),1e-12))
    # The ballistic mean passage is fixed by the Mars entry state: by default the
    # stochastic solve keeps the reoptimised CA and Mars-exit epochs.
    robust=PsycheNLP(data,options,seed=seed,covariance_std=std,
                     mars_days=None if free_mars_events else seed['times'])
    z,report=robust.solve('psyche_stochastic',solver_overrides=snopt_overrides)
    np.savez_compressed(output/'solution.npz',z=z,**robust.unpack(z),covariance_std=std,gain_scale=seed['gain_scale'])
    write(output/'result.json',report)
    print(json.dumps(report),flush=True)
    postprocess(output,samples)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--substeps',type=int,default=1)
    p.add_argument('--mesh-refinement',type=float,default=MESH_REFINEMENT,
                   help='equal numerical intervals per deterministic energy-mesh interval')
    p.add_argument('--stage',choices=['nominal','all','diagnostics'],default='all')
    p.add_argument('--samples',type=int,default=50000)
    p.add_argument('--nominal-feasibility-tolerance',type=float,
                   help='Optional stricter SNOPT feasibility tolerance for energy reoptimisation only')
    p.add_argument('--reduced-hessian-dimension',type=int,
                   help='Optional SNOPT workspace experiment; no change to mathematical constraints')
    p.add_argument('--objective',choices=('standard','quantile'),default='standard',
                   help='quantile: minimize the duration-weighted thrust-chance quantile')
    p.add_argument('--free-mars-events',action='store_true',
                   help='Keep the CA and Mars-exit epochs as stochastic variables (default: fixed)')
    args=p.parse_args()
    result=run(**vars(args))
    raise SystemExit(0 if result['converged'] else 1)
