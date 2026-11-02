# Psyche (multiphase, ballistic Mars)

Psyche runs through the same pipeline as the single-phase cases: energy
reoptimisation on the CPU, Bryson TVLQR and UT covariance seed, the shared
stochastic NLP of `src/model.jl` (native ExaModels algebra, the `add_eval` UT
moment oracle of `src/oracles/ut_moments.jl` and the thrust-chance oracle of
`src/oracles/chance.jl`), and MadNLP with the shared CPU (MUMPS) or GPU
(condensed KKT, cuDSS) settings. Only the multiphase layout, the event-constrained
energy reoptimisation and the Mars clearance row are Psyche-specific. Python only
exports the immutable mission arrays (`tools/export_psyche_inputs.py`) and
validates saved solutions (`tools/validate_psyche.py`).

## Files

| File | Content |
|---|---|
| `data.jl` | Mission arrays, Mars ephemeris pieces, uncertainty model, per-phase UT parameters, nominal arc map |
| `nominal.jl` | Energy reoptimisation with free Mars epochs and the encounter events (CPU, ForwardDiff derivatives) |
| `initialization.jl` | Duration-weighted Bryson TVLQR (K = 0 on Mars) and UT covariance seed |
| `geometry.jl` | Moving-axis B-plane, its analytic Jacobian, event map and clearance |
| `model.jl` | Multiphase `StochasticLayout` and the clearance oracle (CPU and CUDA kernels) |
| `solve.jl` | Driver, literal re-evaluation, report and result files |

## Model

* Phases Sun / Mars / Sun from the fixed SPICE Earth SOI exit to the asteroid
  rendezvous; the Mars SOI entry state is fixed. One XMDS2 RK9 step per interval
  (UT dimension 16, as in the single-phase cases).
* Mars is ballistic: no thrust, no mass flow, no feedforward or gain variables. Its
  UT group propagates one executed point per outer sigma point instead of six
  identical Gates points.
* The energy reoptimisation keeps the closest-approach and Mars-exit epochs free
  (r.v = 0 at the closest-approach node, SOI-exit radius, outward crossing,
  0.05-10 day passage, q > 0, C3 > 0, altitude >= 500 km at the Mars nodes). The
  stochastic NLP fixes them, so every interval duration, cost weight and origin
  translation is a constant: the translations are constants in the two switch arcs'
  mean-defect rows (right-hand sides 0, so the GPU's relaxed equalities keep a 1e-8
  band), and the switch arcs' mean rows carry the scale ratio.
* Cost weights w_k = N dt_k / (t_f - t_0) on every term of the shared objective;
  TVLQR uses Q_k = w_k Q and R_k = w_k R.
* Uncertainty: 2500 km per position axis (`--initial-position-std-km`), the launch
  C3/RA/DEC covariance of the export in velocity, 0.001 kg in mass; navigation
  200 km / 0.1 m/s, acceleration diffusion and Gates errors as Earth-Mars.
  Covariances are normalized by the initial standard deviations; the factor variable
  of node k is L_k with P_k = S_k L_k L_k' S_k', S_k = chol(seed P_k) (full node scales,
  every seed factor the identity).
* Constraints: fixed boundary states (19 rows), UT mean and covariance defects,
  terminal Loewner bound diag((1500 km)^2 I3, (2 m/s)^2 I3)
  (`--psyche-terminal-position-std-km`, `--psyche-terminal-velocity-std-km-s`) or, with
  `--terminal-covariance=cost`, the Bryson term w_N tr(G P_N G')/c^2 in its place,
  thrust chance on the 101 cruise arcs, mean mass >= 1662 kg, and the B-plane
  clearance at the closest-approach node: ||b||/rho - 1 - chi sqrt(lambda_max(C_b))/rho >= 0
  with rho = R_Mars + 500 km and chi = sqrt(-2 ln 0.001).

## Converged configuration (2026-10-31)

With the default terminal bound every variant stalls (the TVLQR seed violates the thrust budget
on the first arc after Mars and the bound needs that feedback). The configuration that converges
on the CPU and the GPU keeps the same initialization (TVLQR seed at w_P = 1, w = 1000):

    run.jl --case=psyche --mode=cpu|gpu --terminal-covariance=cost --covariance-defect=factor \
        --control-norm-eps=1e-3 --state-covariance-weight=0.0135 --control-covariance-weight=60 \
        --psyche-seed-state-covariance-weight=1 --psyche-seed-control-covariance-weight=1000

* `--terminal-covariance=cost`: the Bryson term in place of the bound (the solution reaches
  ~545 km / 0.56 m/s per axis, inside the 1500 km / 2 m/s target).
* `--covariance-defect=factor`: rows L_{k+1} - chol(V_k), linear in L_{k+1}; the product rows
  L L' - V throw tangential steps off the constraints (dL dL'). On the GPU the UT oracle stays
  on the device and the per-arc Cholesky transforms run on the host.
* `--control-norm-eps=1e-3` (also the mass-flow smoothing): the optimum coasts on ~26 arcs, where
  ||f||_eps and the sigma-point mass flow have curvature 1/eps; 1e-5 stalls the barrier.
* Weights: 92.0 / 2.36 / 5.47 % (mean control / state covariance / control covariance) plus
  0.16 % terminal.

Results: CPU 634 iterations, 0 restoration iterations, 306 s; GPU 1666 iterations, 5 restoration
iterations (longest run 2), 615 s; same objective (43.279) and statistics; Monte Carlo (50 000)
matches the UT terminal standard deviations, no unsafe sample. The GPU's relaxed equalities
(1e-8 in node units, ~1 AU for the heliocentric phases) leave a ~5 km mean mismatch in the
independent re-propagation; the CPU solution matches to < 1 mm.

### Without the norm smoothing: polar controls

`--control-parameterization=polar` (same flags otherwise) makes the feedforward variables
(Gamma_k, alpha_k, beta_k) with S_k = Gamma_k d_k, Gamma_k >= 0 and d_k = H_k (cos a cos b, sin a cos b, sin b),
H_k the reflection taking e1 to the seed direction (the start (||S_seed||, 0, 0) is the seed itself).
||S_k|| = Gamma_k in the objective and the chance rows; the UT model uses the plain mass-flow norm and Gates
factors without jitter, so the executed points of a command without feedback are Gamma (d +- sqrt(3) G_l(d))
and their mass flow Gamma ||d +- sqrt(3) G_l(d)|| is smooth at Gamma = 0. No eps or delta remains in the model
(`--control-norm-eps` then only sets the nominal and seed solves, which are unchanged). Gamma >= 0 is placed
so that MadNLP's relaxed bound is exactly 0 (CPU: no relaxation; GPU: `condensed_relaxation`).

CPU (2026-10-31): SOLVE_SUCCEEDED in 413 iterations, 0 restoration iterations, 207 s (eps = 1e-3: 634, 0,
306 s); objective 43.2424 (43.2792, of which 0.034 was the smoothing); propellant 750.27 kg; same shares and
terminal statistics; 41 arcs coast with |S| ~ 4e-8 (the barrier limit; with eps, 28 arcs at |S| ~ 1e-4).
Independent JAX check (with the model's zero jitter): means 0.5 mm, covariances 6e-11; Monte Carlo 0/50 000
unsafe. Not run on the GPU yet.

### Throttle and direction, seed at the cost weights, >= 95 % mean control

    run.jl --case=psyche --mode=cpu --terminal-covariance=cost --covariance-defect=factor \
        --control-norm-eps=1e-3 --state-covariance-weight=0.006 --control-covariance-weight=32 \
        --control-parameterization=throttle --constr-mult-init-max=1e4

* `--control-parameterization=throttle`: the deterministic optimisation's controls, throttle Gamma_k in [0, 1]
  and direction d_k with rows ||d_k||^2 - 1 = 0; S_k = Gamma_k d_k, ||S_k|| = Gamma_k (no eps). The UT and chance
  oracles take Gamma_k as one more input (57 and 43; `ThrottleControls`), CPU only.
* The TVLQR seed uses the cost weights (no `--psyche-seed-*` flags).
* `--constr-mult-init-max=1e4`: MadNLP's least-squares initial multipliers (max ~2e3 here) are kept. With its
  default 1e3 they are all set to zero, the first step sees only the objective's curvature, and from this seed it
  moves the post-Mars factors (arcs 111-117) by up to 15 (covariance rows violated by 24 after one step); the
  solve then stalls and fails in restoration (polar and throttle alike). With the multipliers kept the first step
  violates the rows by 1e-5.

CPU (2026-10-31): SOLVE_SUCCEEDED in 385 iterations, 0 restoration iterations, 194 s; shares 95.55 / 1.27 /
3.07 / 0.11 %; objective 41.557; terminal std 452 / 451 / 457 km, 0.41-0.42 m/s; largest thrust budget 0.827;
propellant 749.45 kg; 42 coast arcs (Gamma ~ 3e-8); | ||d|| - 1 | <= 2e-13. Independent JAX check: means 0.4 mm,
covariances 3e-11; Monte Carlo 0/50 000 unsafe.

### Doubled cruise mesh

`tools/export_psyche_inputs.py --cruise-subdivision 2 --output data/psyche_inputs_cruise2.npz` splits every
solar-cruise interval of the 1.5-refined mesh in two (112 / 54 / 90 arcs, 202 powered; Mars unchanged); runs use
`--psyche-input-file=data/psyche_inputs_cruise2.npz` and the validator rebuilds the same mesh from the report.

* Same flags as above (sphere rows): the solve stalls at iteration ~1060 (objective 67.426 frozen, steps ~2e-4
  after 12-13 backtracks) and then enters repeated restoration (102 iterations by 1538; stopped). The 86 coast arcs
  leave their directions free on the sphere, Newton steps move them tangentially and the quadratic rows
  ||d||^2 = 1 are violated by |dd|^2, so the line search cuts every step.
* With `--direction-constraint=ball` (rows ||d||^2 <= 1): SOLVE_SUCCEEDED in 524 iterations, 0 restoration
  iterations, 522 s (2026-10-31). Lossless: | ||d|| - 1 | <= 6e-7 on the 116 thrusting arcs; the 86 coast arcs
  (Gamma ~ 1.5e-8) keep interior directions (||d|| 0.15-0.78); model objective = literal objective to 2e-6.
  Shares 97.11 / 0.64 / 2.22 / 0.03 %; terminal std 312 / 312 / 320 km, 0.31-0.32 m/s; largest budget 0.874;
  propellant 747.47 kg; JAX check means 0.7 mm, covariances 5e-8 (the solve's own violation); Monte Carlo
  0/50 000 unsafe. The finer mesh also doubles the feedback update rate, hence the smaller dispersions.

### Mesh refined by 2 w.r.t. the energy reference, deterministic-form energy reoptimisation

`--mesh-refinement 2` export (`data/psyche_inputs_refine2.npz`: 74 / 72 / 60 arcs, 134 powered). With
`--control-parameterization=throttle` the energy reoptimisation and the seed use the deterministic optimisation's
formulation: throttle Gamma in [0, 1], direction d with ||d||^2 = 1, thrust T_max Gamma d, mass flow T_max Gamma,
energy 0.5 sum dt Gamma^2 ||d||^2 / (t_f - t_0), no smoothing (and the seed's UT propagation uses the exact
model parameters); `--control-norm-eps` is then unused.

    run.jl --case=psyche --mode=cpu --psyche-input-file=data/psyche_inputs_refine2.npz --terminal-covariance=cost \
        --covariance-defect=factor --state-covariance-weight=0.006 --control-covariance-weight=32 \
        --control-parameterization=throttle --direction-constraint=sphere --constr-mult-init-max=1e4

CPU (2026-10-31): energy reoptimisation 7 iterations (6.1616e5 N^2 s); stochastic SOLVE_SUCCEEDED in 529
iterations, 0 restoration iterations, 325 s, exact sphere rows (| ||d|| - 1 | <= 1e-7); shares 96.66 / 0.84 / 2.46 /
0.04 %; terminal std 305 / 304 / 311 km, 0.30-0.31 m/s; largest budget 0.695; propellant 747.61 kg; 57 of 134
powered arcs coast; JAX check means 0.6 mm, covariances 5e-11; Monte Carlo 0/50 000 unsafe.

## Acceptance

`converged` requires MadNLP `SOLVE_SUCCEEDED`, no restoration run longer than five
iterations and a hyperbolic closest-approach minimum (q > 0, C3 > 0 at the solved
mean). The report also gives the iteration and total restoration counts, the cost
shares, the literal residuals, the terminal and encounter statistics.

## Verification

* `test/psyche_tests.jl`: uncertainty model, ballistic Mars, the analytic B-plane
  Jacobian against ForwardDiff, the model rows against the literal arc map, and
  directional central differences of the Jacobian and the Lagrangian Hessian.
* `test/psyche_gpu_tests.jl` (`--gpu`): CUDA values and derivatives against the CPU model.
* `tools/validate_psyche.py`: independent JAX evaluation of a saved solution,
  Monte Carlo with sample-specific SOI crossings and closest approaches, figures.

### Fuel-optimal reference, 98 % mean control

`tools/export_psyche_inputs.py --reference output/psyche_mission_deterministic_traj_opt/fuel_optimal/psyche.npz
--mesh-refinement 2 --output data/psyche_inputs_fuel_refine2.npz` (80 / 172 / 80 arcs, 160 powered; the starting
throttle is not capped: 34 arcs at full throttle) with `--psyche-nominal-objective=fuel` (the deterministic
1 - m_f / m_wet with the throttle controls) and the throttle flags above, w_P = 0.0034, w = 18.1 in the cost and the
TVLQR seed: SOLVE_SUCCEEDED in 787 iterations, 0 restoration iterations, 581 s (2026-10-31); shares 98.10 / 0.43 /
1.44 / 0.02 %; terminal std 275 / 275 / 281 km, 0.26-0.28 m/s; propellant 747.87 kg; Monte Carlo 0/50 000 unsafe.
With w_P = 0.006, w = 32 the same start converged in 643 iterations but with 71 restoration iterations (one run).
In both the profile is not bang-bang (largest throttle 0.75-0.82, thrust chance never active), close to the
energy-start solution: the stochastic terms, not the start, spread the thrust.

