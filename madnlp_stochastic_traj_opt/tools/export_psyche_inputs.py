"""Export immutable mission data for the native Julia/ExaModels Psyche model.

This is input preparation only: no optimisation, TVLQR, moment propagation or
derivative callback is performed in Python. Native Julia owns those operations.
"""
import argparse
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np
from scipy.interpolate import PPoly
from stochastic_optimization.psyche_inputs import PsycheInput, MESH_REFINEMENT


def export(output,reference=None,mesh_refinement=MESH_REFINEMENT,cruise_subdivision=1):
    data = PsycheInput(reference,mesh_refinement,cruise_subdivision=cruise_subdivision)
    mission = data.mission
    spline = mission.cache._splines[499]
    polynomials = [PPoly.from_spline((spline.t,spline.c[:,j],spline.k)) for j in range(6)]
    scales = np.array([s.state for s in mission.scales])
    # Numeric arrays only, with explicit axis conventions for Julia's NPZ loader.
    arrays = dict(reference_path_utf8=np.frombuffer(str(data.path.resolve().relative_to(ROOT)).replace('\\','/').encode(),dtype=np.uint8),
        reference_sha256_ascii=np.frombuffer(hashlib.sha256(data.path.read_bytes()).hexdigest().encode(),dtype=np.uint8),
        seed_states=data.seed_states.T,seed_controls=data.seed_controls.T,
        seed_durations=data.seed_durations,phase_state_scales=scales.T,
        phase_time_scales=np.array([s.time for s in mission.scales]),
        phase_gravitational_parameters=np.array([s.mu for s in mission.scales]),
        node_scales=data.node_scales.T,node_origins=np.array(data.node_origins,dtype=np.int64)+1,
        arc_phase=np.array([a.phase for a in data.arcs],dtype=np.int64)+1,
        arc_start=np.array([a.start for a in data.arcs],dtype=np.int64)+1,
        arc_end=np.array([a.end for a in data.arcs],dtype=np.int64)+1,
        arc_switch=np.array([a.switch for a in data.arcs],dtype=np.int64),
        arc_duration=np.array([a.duration for a in data.arcs]).T,
        initial_covariance=data.initial_covariance,initial_covariance_si=data.initial_covariance_si,
        covariance_scale=data.covariance_scale,navigation_covariance=data.navigation_covariance,
        terminal_inverse_std=data.terminal_inverse_std,
        mars_entry_node=data.mars_entry_node+1,mars_exit_node=data.mars_exit_node+1,
        closest_approach_node=data.ca_node+1,ballistic_mars=np.array(1, dtype=np.int64),
        arc_control_index=data.control_indices+1,
        fixed_epochs=np.array([mission.t0,mission.tm,mission.tf]),
        x0=mission.x0,xm=mission.xm,xf=mission.xf,
        thrust_max_n=mission.settings.thrust_max_n,isp_s=mission.settings.isp_s,
        wet_mass_kg=mission.settings.wet_mass_kg,mass_floor_kg=mission.settings.mass_floor_kg,
        mars_radius_m=mission.settings.mars_radius_m,mars_soi_m=mission.soi[499],
        ephemeris_knots=polynomials[0].x,
        ephemeris_coefficients=np.stack([p.c for p in polynomials],axis=-1),
        ephemeris_origin_et=mission.cache.origin_et,ephemeris_valid_interval=mission.cache.valid_interval,
        phase_mesh_counts=np.array(data.phase_mesh_counts,dtype=np.int64),
        reference_phase_mesh_counts=np.array([len(p['mesh'])-1 for p in data.solution['phases']],dtype=np.int64),
        mesh_refinement=np.array(data.mesh_refinement,dtype=np.float64),
        cruise_subdivision=np.array(data.cruise_subdivision,dtype=np.int64))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    report = dict(reference=str(data.path),reference_sha256=hashlib.sha256(data.path.read_bytes()).hexdigest(),
        export_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        phase_mesh=arrays['phase_mesh_counts'].tolist(),reference_phase_mesh=arrays['reference_phase_mesh_counts'].tolist(),
        mesh_refinement=data.mesh_refinement,cruise_subdivision=data.cruise_subdivision,
        control_arcs=data.n_powered_controls,
        numerical_intervals=data.n_intervals,
        position_units='m',velocity_units='m/s',
        time_units='TDB seconds past J2000; free segment durations in days',orientation='J2000',
        preparation_only=True,optimisation_backend='native Julia/ExaModels',
        state_array_order='components x nodes',ephemeris_array_order='polynomial coefficient x interval x component')
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--mesh-refinement',type=float,default=MESH_REFINEMENT,
                        help='equal numerical intervals per deterministic energy-mesh interval')
    parser.add_argument('--cruise-subdivision',type=int,default=1,
                        help='then split every solar-cruise interval into this many equal pieces (Mars mesh unchanged)')
    export(**vars(parser.parse_args()))
