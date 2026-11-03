# Psyche (multiphase, ballistic Mars)

Psyche runs through the same pipeline as the single-phase cases: fuel reoptimisation on the CPU, Bryson TVLQR and
UT covariance seed, the shared stochastic NLP of `src/model.jl` (native ExaModels algebra, the UT moment and factor
oracles and the thrust-chance oracles of `src/oracles/`), and MadNLP with the shared settings, for both problems
(step 3 and step 4, `../../README.md`). Only the multiphase layout, the event-constrained reoptimisation and the
Mars clearance row are Psyche-specific. Python only exports the immutable mission arrays
(`tools/export_psyche_inputs.py`) and validates saved solutions (`tools/validate_psyche.py`).

## Files

| File | Content |
|---|---|
| `data.jl` | Mission arrays, Mars ephemeris pieces, uncertainty model, per-phase UT parameters, nominal arc maps |
| `nominal.jl` | Fuel reoptimisation with free Mars epochs and the encounter events (CPU, ForwardDiff derivatives) |
| `initialization.jl` | Duration-weighted Bryson TVLQR (K = 0 on Mars) and UT covariance seed |
| `geometry.jl` | Moving-axis B-plane, its analytic Jacobian, event map and clearance |
| `model.jl` | Multiphase `StochasticLayout` and the clearance oracle (CPU and CUDA kernels) |
| `solve.jl` | Driver, literal re-evaluation, report and result files |

## Model

* Mission arrays: `data/psyche_inputs_fuel_refine2.npz`, exported from the deterministic fuel-optimal solution with
  every interval split in two (`tools/export_psyche_inputs.py --reference
  output/psyche_mission_deterministic_traj_opt/fuel_optimal/psyche.npz --mesh-refinement 2 --output
  data/psyche_inputs_fuel_refine2.npz`): 80 / 172 / 80 arcs in the Earth–Mars, Mars and Mars–Psyche phases,
  160 powered.
* Phases Sun / Mars / Sun from the fixed SPICE Earth SOI exit to the asteroid rendezvous; the Mars SOI entry state
  is fixed. One XMDS2 RK9 step per interval (UT dimension 16, as in the single-phase cases).
* Mars is ballistic: no thrust, no mass flow, no throttle, direction or gain variables. Its UT group propagates
  one executed point per outer sigma point instead of six identical Gates points.
* Controls: the deterministic optimisation's throttle Gamma_k in [0, 1] and unit direction d_k (thrust Gamma d, mass
  flow T_max Gamma in the reoptimisation and the seed).
* The fuel reoptimisation (objective 1 - m_f / m_wet) keeps the closest-approach and Mars-exit epochs free
  (r.v = 0 at the closest-approach node, SOI-exit radius, outward crossing, 0.05-10 day passage, q > 0, C3 > 0,
  altitude >= 500 km at the Mars nodes). The stochastic NLP fixes them, so every interval duration, cost weight and
  origin translation is a constant: the translations are constants in the two switch arcs' mean-defect rows, and
  the switch arcs' mean rows carry the scale ratio.
* Cost weights w_k = N dt_k / (t_f - t_0) on every term of the objective; TVLQR uses Q_k = w_k Q and R_k = w_k R.
* Uncertainty: 2500 km per position axis, the launch C3/RA/DEC covariance of the export in velocity, 0.001 kg in
  mass; navigation 200 km / 0.1 m/s, acceleration diffusion and Gates errors as Earth–Mars. Covariances are
  normalized by the initial standard deviations; the factor variable of node k is L_k with P_k = S_k L_k L_k' S_k',
  S_k = chol(seed P_k) (full node scales, every seed factor the identity).
* Tracking weights (step 3): w_P = 0.0102, w = 54.3, a 95 % mean-control share of the cost.
* Constraints: fixed boundary states (19 rows), UT mean and covariance defects, the terminal covariance target
  diag((1500 km)^2 I3, (2 m/s)^2 I3) (step 3: the Bryson term w_N tr(G P_N G')/c^2 in the cost; step 4: the hard
  bound), thrust chance on the powered arcs, mean mass >= 1662 kg, and the B-plane clearance at the closest-approach
  node: ||b||/rho - 1 - chi sqrt(lambda_max(C_b))/rho >= 0 with rho = R_Mars + 500 km and chi = sqrt(-2 ln 0.001).

## Acceptance

`converged` requires MadNLP `SOLVE_SUCCEEDED`, no restoration run longer than five iterations and a hyperbolic
closest-approach minimum (q > 0, C3 > 0 at the solved mean). The report also gives the iteration and total
restoration counts, the cost shares, the literal residuals, the terminal and encounter statistics.

## Verification

* `test/psyche_tests.jl`: uncertainty model, ballistic Mars, the analytic B-plane Jacobian against ForwardDiff, both
  models' rows against the literal arc map, and directional central differences of the Jacobian and the Lagrangian
  Hessian.
* `test/psyche_gpu_tests.jl` (`--gpu`): CUDA values and derivatives against the CPU model.
* `tools/validate_psyche.py`: independent JAX evaluation of a saved solution, Monte Carlo (50 000 samples) with
  sample-specific SOI crossings and closest approaches, figures.
