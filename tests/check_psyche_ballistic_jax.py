"""Ballistic-Mars checks for the JAX maps; run in the JAX environment.

No optimisation is run. Requires the deterministic energy reference for the
Monte Carlo integrator check.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from deterministic_optimization.psyche import Mission
from stochastic_optimization.psyche_geometry import launch_covariance_si
from stochastic_optimization.psyche_inputs import Arc, PsycheInput
from stochastic_optimization.psyche_maps import Maps
from stochastic_optimization.psyche_monte_carlo import ensemble_integrator
import stochastic_traj_opt_jax as common

m=Mission()
d=SimpleNamespace(mission=m,covariance_scale=np.sqrt(np.diag(launch_covariance_si())))
d.duration=lambda arc,days: arc.duration[0]+jnp.asarray(arc.duration[1:])@days
d.transform_end=lambda arc,x,days:x
o=replace(common.default_options('earth_mars'),integrator_substeps=2,sde_integrator_substeps=2)
maps=Maps(d,o)
arc=Arc(1,0,1,0,np.array([100.,0.,0.]))
y=jnp.asarray(np.r_[m.xm,2400.]/m.scales[1].state)
f=jax.jit(lambda u,k:maps.stochastic(arc,y,jnp.eye(7),u,k,jnp.ones(2)))
z=f(jnp.zeros(3),jnp.zeros((3,6)))
w=f(jnp.ones(3),jnp.ones((3,6)))
for a,b in zip(z,w):
    np.testing.assert_array_equal(a,b)
    assert np.isfinite(a).all()
# Weighted sigma-point summation introduces roundoff although each point has
# exactly constant mass (228 nested points with the retained UT quadrature).
np.testing.assert_allclose(np.asarray(z[0])[6],y[6],atol=1e-14,rtol=0)
assert np.all(np.asarray(z[2])==0)
derivative=jax.jit(jax.jacrev(lambda state: maps.stochastic(
    arc,state,jnp.eye(7),jnp.zeros(3),jnp.zeros((3,6)),jnp.ones(2))[0]))(y)
assert np.isfinite(derivative).all()
print('JAX ballistic moments: zero control/gain dependence, no execution covariance, conserved mean mass; PASS',flush=True)

# Monte Carlo: an engine-off (ballistic Mars) command consumes no propellant,
# while a cruise command uses the NLP's smoothed thrust norm.
data=PsycheInput()
advance=ensemble_integrator(data,sample_specific=False)
count=4
state=np.tile(np.r_[m.xm+m.cache.state(499,m.tm),2400.],(count,1))
args=(jnp.zeros((count,3)),m.tm,600.,jnp.ones(count,dtype=jnp.int32),jnp.full(count,jnp.inf),
      jnp.full(count,m.tm),jnp.asarray(state),jnp.full(count,jnp.nan),jnp.full(count,jnp.nan))
coast=np.asarray(advance(jnp.asarray(state),jnp.zeros((count,3)),*args)[0])
np.testing.assert_array_equal(coast[:,6],state[:,6])
burn=np.asarray(advance(jnp.asarray(state),jnp.full((count,3),.1),*args)[0])
expected=600.*np.sqrt(3*.1**2+(m.settings.thrust_max_n*1e-5)**2)/(m.settings.isp_s*9.80665)
# Mass is ~2400 kg, so the difference carries ~1e-13 kg of roundoff.
np.testing.assert_allclose(state[:,6]-burn[:,6],expected,rtol=0,atol=1e-12)
print('Monte Carlo mass flow: zero for ballistic commands, smoothed norm otherwise; PASS',flush=True)
