# Psyche stochastic optimisation (ballistic Mars)

Chance-constrained covariance steering for the three-phase Psyche transfer
(Sun, Mars, Sun). Two implementations share the same formulation:

* **JAX/SNOPT** — this package, driven by `stochastic_traj_opt_jax.py --case psyche`.
  It is also the independent reference used to verify native solutions.
* **Native Julia/ExaModels/MadNLP** — `madnlp_stochastic_traj_opt/src/psyche/`
  (CPU and GPU), fed by the mission arrays exported from this package.

The formulation follows the plan `data/psyche_sources/psyche_stochastic_plan.pdf`
(SHA256 `e88ddc55ec0b4c009cc768e01adaf7b89d44416f6800f7878a9edbb3e3174674`), with
the later approved revisions recorded below: executed-control chance covariance,
ballistic Mars, unsmoothed B-plane eigenvalue, two RK9 substeps, and a hard
500 km Mars altitude in the RK9 energy reoptimisation.

## Dynamics, boundaries and variables

* Start at the fixed reconstructed Earth SOI-exit state/epoch, fix the Mars SOI
  entry state/epoch, and rendezvous with the asteroid centre at
  2029-08-01 12:00 TDB. The Mars SOI is the fixed-semimajor-axis Laplace radius;
  the Mars exit is free (outward crossing, 0.05–10 day Mars duration).
* Point-mass central gravity per phase; origin translations use the stored
  differentiable cubic ephemerides. No SPICE call is made during optimisation.
* The 2747 kg initial mass at Earth SOI exit is a modelling assumption.
* **Mars passage is ballistic**: T = 0, K = 0 and dm/dt = 0 exactly. Control and
  gain variables exist only on cruise arcs. Smoothing and execution jitter cannot
  produce a Mars burn.
* The mesh is the accepted deterministic energy mesh (currently 37/36/30 arcs)
  plus the closest-approach (CA) node, which splits a Mars interval or reuses a
  coincident node (the current mesh reuses one). Inbound and outbound Mars
  intervals keep their fractions of the free CA epoch. CA is an ordinary
  state/covariance node: no reset and no new command.
* With N intervals and C cruise commands the stochastic NLP has
  7(N+1) + 3C + 18C + 28N + 21 variables: node means, feedforward commands,
  scaled gains, node covariance factors (the initial covariance is fixed) and
  the terminal-margin factor.

## Mars event epochs

The Mars passage is ballistic and starts from a fixed state, so the nominal CA
and SOI-exit epochs are fixed by the entry state. The two Mars durations
(entry → CA → exit) are therefore handled differently in the two NLPs:

* **Energy reoptimisation:** variables, with CA r·v = 0, SOI-exit radius,
  outward-crossing and 0.05–10 day duration rows. Its solution supplies the
  epochs (they are the CA and exit roots of the RK9-discretised trajectory
  driven by the reoptimised control history).
* **Stochastic NLP (default):** fixed at those epochs. Interval durations, the
  exit translation and the cost weights become constants; the CA, SOI-exit,
  outward-crossing and duration rows are dropped. q > 0 and C3 > 0 are checked
  after solving, and the event rows of the solved mean are reported. The B-plane
  quantities are constants of the Mars-only ballistic flow, so the clearance
  constraint at the CA node does not depend on the exact CA epoch.
* `--psyche-free-mars-events` (native and `stochastic_traj_opt_jax.py`;
  `--free-mars-events` in `psyche_run`) keeps the durations as stochastic
  variables with all event rows, for comparison.

At equal epochs the fixed and free models give identical values and
derivatives (JAX exactly; native up to 8e-17 in the Mars-exit defect rows,
from summation order). Fixing removes two dense columns: about 7 300 Jacobian
and 10 000 Hessian nonzeros in the native model.

## Uncertainty, cost and constraints

* Initial covariance: 2000 km per position axis; asymptotic C3/RA/DEC bounds read
  as independent 3σ values and mapped through the velocity Jacobian (correlations
  retained, applied at Earth SOI as an approximation); 0.001 kg in mass.
* Navigation 200 km / 0.1 m/s, acceleration diffusion 1e-7 m/s^(3/2), proportional
  Gates execution errors 0.008 (magnitude) and 0.0095993 rad (pointing), as in
  Earth–Mars. The nested UT forms each outer point's conditional execution
  covariance from its own command. With two RK9 substeps the UT dimension is 19.
* Cruise thrust chance constraints (risk 0.01) use the **executed-control**
  covariance: commanded dispersion including navigation feedback, plus execution
  error, each once.
* Cost, with interval weights w_k = N·dt_k/T_total:

      sum_cruise w_k (sqrt(f_k'f_k + 1e-10) + trace(C_executed,k))
        + sum_all   w_k trace(P_k[0:6, 0:6]) / 9

* Mean mass ≥ 1662 kg at nodes. No mean surface or SOI path constraints in the
  stochastic NLP.
* Terminal covariance: full Loewner bound with 2000 km / 2 m/s per-axis scales.
* Closest approach: q = v·v + r·a > 0 and C3 > 0 (a valid hyperbolic local
  minimum) are checked after solving; r·v = 0 is a row only with free Mars
  events (see above). The B-plane
  covariance uses moving S/T/R axes, focusing and the event map
  E = I − [v; a][v; r]ᵀ/q at the common epoch. Altitude ≥ 500 km with risk 0.001.
  The largest B-plane covariance eigenvalue is **unsmoothed**; it is
  nondifferentiable at repeated eigenvalues and no floor is substituted.

## Initialisation and integration

1. **RK9 energy reoptimisation** from the deterministic energy reference, with the
   CA and Mars-exit epochs initialised from the deterministic trajectory and kept
   free; its epochs are then fixed for the stochastic NLP. It enforces Mars altitude ≥ 500 km at Mars nodes and at every uniform RK9
   step endpoint (no Sun exclusion, no B-plane constraint).
2. **Duration-weighted Bryson TVLQR** (Q = I/9, R = I, Psyche's terminal weight
   from the 2000 km / 2 m/s limits). Mars arcs keep K = 0 but propagate the
   Riccati matrix, so cruise feedback anticipates the uncontrolled flyby.
3. UT covariance propagation of the seed, then the terminal-margin factor.
4. The stochastic NLP.

The default is two uniform RK9 substeps per interval in all stages
(`--psyche-substeps` / `--substeps`). Diagnostic comparisons use twice as many.

## Commands

From the repository root:

```bash
# Deterministic references and the native input export
python deterministic_traj_opt.py --case psyche --objective both
python madnlp_stochastic_traj_opt/tools/export_psyche_inputs.py --output madnlp_stochastic_traj_opt/data/psyche_inputs.npz

# JAX/SNOPT (SNOPT licence and helper library required; see wsl_snopt_python.sh)
JAX_PLATFORMS=cuda python stochastic_traj_opt_jax.py --case psyche
python -m stochastic_optimization.psyche_run --stage diagnostics --output PATH_TO_SAVED_RUN

# Native CPU/GPU
julia --project=madnlp_stochastic_traj_opt --threads=4 madnlp_stochastic_traj_opt/run.jl --case=psyche --mode=cpu --max-iter=10000 --nominal-max-iter=10000
python madnlp_stochastic_traj_opt/tools/validate_psyche.py madnlp_stochastic_traj_opt/output/cpu/psyche
```

`wsl_snopt_python.sh` runs the WSL `project` Python with the installed SNOPT
helper preload and licence path, e.g.
`bash stochastic_optimization/wsl_snopt_python.sh stochastic_traj_opt_jax.py --case psyche`.

The native export is tied to the deterministic energy NPZ by SHA256; regenerate
it whenever that reference changes. JAX outputs default to
`output/psyche_mission_stochastic_traj_opt_jax/{cpu,gpu}`.

Acceptance (JAX): SNOPT INFO 1, finite values, valid CA domain and original
constraint violation ≤ the major feasibility tolerance (1e-8). Native acceptance
is MadNLP `SOLVE_SUCCEEDED`, the restoration-run requirement and the CA domain;
its original residuals are reported, not gated. Only accepted JAX solutions
launch the post-processing ensembles: three 50 000-sample Monte Carlo runs
(seeds 42–44) and a coupled finer-integration run.

## Diagnostics

* `physical_diagnostics`: physical states/covariances, residuals in SI, cost
  breakdown, B-plane encounter statistics and SOI vector-field mismatch scales.
* `moment_accuracy`: the NLP moment map against twice as many RK9 steps, either
  with identical UT points and prescribed Wiener paths, or with an independent
  finer Wiener/UT grid (not a pure ODE error). Includes the change in the CA
  clearance on the arc that ends at CA.
* `nominal_linearization_accuracy`: nominal Jacobians and recomputed TVLQR gains
  at two step counts on a fixed path.
* Monte Carlo (`psyche_monte_carlo.py`): continuous RK9 ensembles switching
  gravity at each sample's own SOI crossings (a different hybrid model from the
  scheduled NLP switches), with bracketed CA and SOI roots, no resets and no
  extra draws at evaluation nodes. Ballistic commands consume no propellant.
  Binomial intervals describe sampling uncertainty only.

Passive loss-of-thrust scenarios and third-body gravity are not implemented.

## Modules

| Module | Purpose |
|---|---|
| `psyche_inputs.py` | Validated deterministic reference, CA-augmented mesh, stored ephemeris splines |
| `psyche_geometry.py` | Launch covariance; analytic and AD moving-axis B-plane Jacobians and encounter statistics |
| `psyche_maps.py` | RK9 nominal and UT moment maps per interval |
| `psyche_nlp.py` | Sparse SNOPT transcription (energy reoptimisation and stochastic NLP) |
| `psyche_run.py` | Energy reoptimisation → TVLQR → stochastic solve → post-processing |
| `psyche_diagnostics.py` | Physical diagnostics and integration-accuracy studies |
| `psyche_monte_carlo.py` | Continuous ensemble with sample-specific events |
| `psyche_plots.py` | Figures in the shared `plotter.py` style |

Checks: `python tests/check_psyche_ballistic_jax.py` (JAX environment; no solver).
