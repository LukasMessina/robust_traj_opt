"""Independent Python verification and shared-style plots of a native Julia solve.

This is post-processing only. It is never an optimizer callback, and does not
change acceptance or initialise another solve from this solution.
The JAX device (Monte Carlo included) follows the solve's execution mode,
recorded in psyche_diagnostics.toml, unless --mode overrides it.
"""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
import tomllib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import jax_backend


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('directory',type=Path)
    result.add_argument('--mode',choices=jax_backend.MODES,help='JAX device (default: the execution mode of the solve)')
    result.add_argument('--no-plots',action='store_true')
    result.add_argument('--samples',type=int,help='Monte Carlo samples (default: as the other cases; 0 skips it)')
    result.add_argument('--seed',type=int,help='Monte Carlo seed (default: as the other cases)')
    result.add_argument('--mc-substeps',type=int,default=16,
                        help='RK9 substeps per interval (at least 16; dividing 32 or a multiple of it)')
    return result


def execution_mode(args):
    if args.mode is not None:
        return args.mode
    return tomllib.loads((args.directory/'psyche_diagnostics.toml').read_text())['execution_mode']


# The device must be chosen before JAX is imported.
ARGS=parser().parse_args() if __name__=='__main__' else None
if ARGS is not None:
    jax_backend.select(execution_mode(ARGS))
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import stochastic_traj_opt_jax as common
from stochastic_optimization.psyche_inputs import PsycheInput
from stochastic_optimization.psyche_nlp import PsycheNLP
from stochastic_optimization.psyche_diagnostics import physical_diagnostics
from stochastic_optimization.psyche_monte_carlo import run_monte_carlo, MONTE_CARLO_MIN_SUBSTEPS
from stochastic_optimization.psyche_plots import plot_diagnostics


def validate(directory,plots=True,samples=None,monte_carlo_seed=None,substeps=MONTE_CARLO_MIN_SUBSTEPS):
    """samples/monte_carlo_seed default to the other cases' Monte Carlo settings."""
    directory=Path(directory)
    report=tomllib.loads((directory/'psyche_diagnostics.toml').read_text())
    if not (directory/'psyche_solution.npz').exists():
        raise ValueError('No native stochastic candidate was saved; see failed_stage in the diagnostics')
    # The run's mesh (runs exported before the refinement option used the energy mesh).
    data=PsycheInput(path=ROOT/report['reference_path'] if 'reference_path' in report else None,
        mesh_refinement=report.get('mesh_refinement',1),cruise_subdivision=report.get('cruise_subdivision',1),
        initial_position_std_m=1000*report.get('initial_position_std_km',2000.),
        terminal_std_si=(1000*report.get('terminal_position_std_km',2000.),1000*report.get('terminal_velocity_std_km_s',.002)))
    if data.n_intervals!=report['n_intervals']:
        raise ValueError(f'The run has {report["n_intervals"]} intervals, the reference mesh {data.n_intervals}')
    settings=dict(report['settings'])
    # TOML stores a non-finite gain bound as the string "Inf" (no gain bound)
    settings['gain_bound']=float(settings['gain_bound'])
    # Polar controls: the model's exact norm (JAX needs a positive eps; 1e-12 is exact to rounding), plain mass
    # flow and Gates factors without jitter (on coast arcs a 1e-12 jitter outweighs Gamma^2 Q_G(d)).
    extra={}
    if 'model_norm_smoothing' in settings:
        settings['control_norm_eps']=max(settings['model_norm_smoothing'],1e-12)
        settings['mass_flow_smoothing']=settings['model_mass_flow_smoothing']
        extra['cholesky_jitter']=settings.get('model_gates_jitter',0.)
    options=replace(common.default_options('earth_mars'),**extra,
        integrator_substeps=report['substeps_per_interval'],sde_integrator_substeps=report['substeps_per_interval'],
        **{k:settings[k] for k in ('control_norm_eps','mass_flow_smoothing','gain_bound','control_covariance_weight',
             'eigenvalue_smoothing','spectral_radius_floor','terminal_margin_floor')},
        objective={'chance_sum':'quantile'}.get(settings.get('objective','standard'),settings.get('objective','standard')),state_covariance_weight=settings.get('state_covariance_weight',1.))
    native=np.load(directory/'psyche_solution.npz')
    saved_seed=np.load(directory/'psyche_seed.npz')
    seed=dict(means=saved_seed['means'].T,feedforward=saved_seed['feedforward'].T,times=saved_seed['days'],
        gains=saved_seed['gains'].transpose(2,0,1),gain_scale=saved_seed['gain_scale'],covariances=saved_seed['covariances'].transpose(2,0,1))
    free=report.get('mars_events','free').startswith('free')
    nlp=PsycheNLP(data,options,seed=seed,covariance_std=native['covariance_std'].T,
                  mars_days=None if free else native['days'])
    z=nlp.x0.copy()
    for key,values in dict(means=native['means'].T,feedforward=native['feedforward'].T,times=native['days'],
        gains=native['gains'].transpose(2,0,1)/native['gain_scale'][:,None,None],factors=native['factors'].T,margin=native['margin']).items():
        if key not in nlp.slices:
            continue
        if key in ('feedforward','gains'):
            values=values[data.powered_arcs]
        z[nlp.slices[key]]=values.ravel()
    arrays,physical=physical_diagnostics(nlp,z)
    physical.update(converged=report['converged'],native_solver_status=report['madnlp_status'],
        native_solver_success=report['solver_success'],restoration_requirement_met=report['restoration_requirement_met'],
        native_objective=report['objective'],independent_objective_difference=physical['objective']-report['objective'],
        qualification='Independent post-solve verification; no residual acceptance gate')
    np.savez_compressed(directory/'psyche_physical_solution.npz',**arrays)
    (directory/'psyche_physical_diagnostics.json').write_text(json.dumps(physical,indent=2,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
    table=np.column_stack((arrays['epochs_et'],data.node_origins,arrays['states_si'],
        np.sqrt(np.maximum(np.diagonal(arrays['covariance_si'],axis1=1,axis2=2),0))))
    np.savetxt(directory/'psyche.csv',table,delimiter=',',comments='',
        header='et_tdb_s,origin_phase,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,m_kg,sx_m,sy_m,sz_m,svx_m_s,svy_m_s,svz_m_s,sm_kg')
    samples=options.monte_carlo_samples if samples is None else samples
    monte_carlo_seed=options.monte_carlo_seed if monte_carlo_seed is None else monte_carlo_seed
    ensembles=[]
    if samples:
        mc,stats=run_monte_carlo(data,arrays,nlp.gain_scale,samples=samples,seed=monte_carlo_seed,substeps=substeps)
        np.savez_compressed(directory/'psyche_monte_carlo.npz',**mc)
        (directory/'psyche_monte_carlo.json').write_text(json.dumps(stats,indent=2)+'\n')
        print(f'Psyche Monte Carlo seed {monte_carlo_seed}: {stats["unsafe_altitude_count"]}/{samples} below 500 km, '
              f'{stats["invalid_hyperbolic_count"]} invalid geometries',flush=True)
        ensembles.append(mc)
    if plots:
        plot_diagnostics(data,arrays,physical,directory,ensembles)
    print(json.dumps({k:physical[k] for k in ('converged','objective','independent_objective_difference',
        'mean_matching_position_m','mean_matching_velocity_m_s','covariance_matching_scaled','terminal_std_si')},indent=2),flush=True)
    return physical


if __name__=='__main__':
    print(f'Checks and Monte Carlo on JAX {jax_backend.check(execution_mode(ARGS))}',flush=True)
    validate(ARGS.directory,plots=not ARGS.no_plots,samples=ARGS.samples,monte_carlo_seed=ARGS.seed,substeps=ARGS.mc_substeps)
