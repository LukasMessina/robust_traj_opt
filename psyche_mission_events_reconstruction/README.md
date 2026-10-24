# Psyche mission inputs and reconstruction

This package reconstructs and verifies the inputs used by the final deterministic optimisation. It does **not** run an optimiser. The validated ten-case batch is retained in [output/deterministic/validated/](../output/deterministic/validated/), with its [saved reference comparison](../output/deterministic/validated/reference_comparison.json). The superseded reference output folders and maintenance archive were removed after validation. The retained inputs in `artifacts/` reproduce that batch; newly reconstructed inputs go into a separate directory.

The four-phase trajectory is an optimisation model informed by mission data. It is not a reproduction of every force, manoeuvre or operational constraint in NASA's flight solution. In particular, the terminal condition is a user-selected rendezvous with the **asteroid centre**, not the spacecraft state from a predicted arrival SPK.

## Reproduce

From the repository root, activate the environment in [environment-deterministic.yml](../environment-deterministic.yml):

```powershell
conda activate det_traj_opt_env
python -m psyche_mission_events_reconstruction
python -m psyche_mission_events_reconstruction --output output/mission_inputs_new --rebuild-cache
python deterministic_traj_opt.py --case psyche --objective both --jobs 2
```

The first command verifies the pinned inputs, kernel hashes, coverage, exact extracted states, C3 and provenance links. The second re-extracts events, rebuilds/validates the planetary cache, and regenerates the coast timing histories. Without `--rebuild-cache`, it reuses the pinned numerical cache and updates the new provenance. An existing reconstruction directory is rejected to protect the reference. The optimiser uses the pinned `artifacts/` by default; set `PSYCHE_RECONSTRUCTION_DATA` to an absolute directory to deliberately use another verified reconstruction. Do not silently replace reference inputs when comparing solver implementations.

Missing kernels can be downloaded explicitly; existing files must match their pinned hashes:

```powershell
python -m psyche_mission_events_reconstruction.download_kernels
```

The optimiser itself has no network or SPICE requirement. TudatPy and SpiceyPy are required for reconstruction; SpiceyPy is also used after solving to orient the textured Earth/Mars figures. Download the plot maps and orientation kernels once with `python -m deterministic_optimization.plot_assets`; their manifest is separate from mission-data inputs. Results are published in `output/psyche_mission_deterministic_traj_opt/{fuel,energy}_optimal/`. On Windows, activating the Conda environment also puts its DLL directory on `PATH`. The original plot style requires a working LaTeX installation and `dvipng`.

## Sources and evidence

* [User's MSc thesis, retained PDF](../data/psyche_sources/MSc_Thesis_Lukas_Messina_Final_Version.pdf): MIS-REQ-01, printed p.15 / PDF p.33 specifies noon **UTC** on 1 August 2029. The user subsequently selected noon **TDB** and the asteroid's state. The thesis spacecraft parameters remain the model parameters below. SHA256: `df86a7b4c0db4626a861ffd303856d4b85019b6be0f4b86d8b6a71a50f6e37e4`.
* [IEPC 2024 paper 126](https://electricrocket.org/IEPC_2024/126.pdf), pp.11–12, Figure 11 and Table 1: post-flight timing and launch injection diagnostics. [Local PDF](../data/psyche_sources/IEPC_2024_126_launch_checkout.pdf), SHA256 `d9757d47f52c0f18ea84323bf5c0cb06fff5512e6d5220ce712e86c8c6150088`.
* [NAIF Psyche archive](https://naif.jpl.nasa.gov/pub/naif/PSYCHE/kernels/spk/): reconstructed launch/Mars states and the selected predicted asteroid ephemeris. Coverage is checked from binary SPK segments, not inferred from filenames or sidecar labels.
* [JPL approximate planetary elements](https://ssd.jpl.nasa.gov/planets/approx_pos.html): Table 1 supplies the frozen Mars semimajor-axis coefficient, 1.52371034 au. Earth uses a nominal 1 au. JPL's 1.00000261 au coefficient is for the Earth–Moon barycentre and is retained only as a sensitivity comparison.
* The historical date/source audit is retained in [evidence/arrival_source_audit](evidence/arrival_source_audit/report.txt). It records the sources reviewed on 23 October 2026, including the [NASA/JPL press kit](https://www.jpl.nasa.gov/press-kits/psyche/mission/), [Polanskey et al. (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12521304/), and [Ermakov (2026)](https://doi.org/10.3847/PSJ/ae7f03). These distinguish capture, science planning dates and an approximate analysis reference epoch. They do not establish this optimisation endpoint as a measured exact capture event.

[evidence_manifest.json](evidence_manifest.json) lists retained source paths, checksums and download URLs. `evidence/initial_thesis_review/` and `evidence/arrival_source_audit/` are historical evidence: their tentative pre-flight TIP and former spacecraft/UTC arrival boundaries are **superseded** by the active events below. The NAIF archive HTML snapshot and reference-SPK labels remain under `data/psyche_sources/`. No covariance or physical uncertainty bound is supplied for the extracted states.

## Kernels and coverage

All files are retained under `data/jpl_kernels/`. Full URLs and SHA256 hashes are also enforced in `kernels.py`, `mars_kernels.py`, `reference_kernel.py` and the evidence manifest.

| Kernel | Role / binary coverage relevant here | SHA256 |
|---|---|---|
| `naif0012.tls` | UTC/TDB conversions; leap-second table, not an SPK | `678e32bdb5a744117a467cd9601cd6b373f0e9bc9bbde1371d5eee39600a039b` |
| `gm_de440.tpc` | Sun/Earth/Mars gravitational parameters | `924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140` |
| `de440s.bsp` | Planetary barycentres, Earth and Sun; ET −4734072000 to 4735368000 s | `c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2` |
| `mar097.bsp` | Mars centre 499 relative to barycentre 4; ET −3155457600 to 3155889600 s | `201fc87075a7aaefc8b4d7f7545881079f906414fae38c5e36b1786f738dca63` |
| `psyche_rec_231013-231018_v1.bsp` | Spacecraft −255; 2023-10-13 15:21:47.957 UTC through 2023-10-18 16:00 UTC | `74f8e2e478df739ce2e6c57f4287079fbefc987b02f7c9350e4567a3ec4c0e10` |
| `psyche_rec_260501-260713_260804_v1.bsp` | Spacecraft −255; 2026-05-01 23:59:59 UTC through 2026-07-13 22:40 UTC | `65a80b4e8d72ac6bdf8177df54525e39de5f812e05c73d7261a6b3e7673476b6` |
| `psyche_ref_231013-311101_231016_v1_merged.bsp` | Asteroid 2000016, Sun-centred type-2 segment; ET 750470469.1823572 to 1004530089.1825076 s | `ac66af58c0e02230e5f652ae323df64c784dc1d4bd242f65fc5aa467779d3958` |

Generic files are downloaded from [NAIF generic kernels](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/); the three Psyche SPKs and `mar097.bsp` use the Psyche archive above. The asteroid segment's source piece is `sb_psyche_ssd_180411.bsp`, identified in the merged kernel comments. Its coverage differs from spacecraft −255 coverage in the same file. This is an older asteroid prediction retained for reproducibility, not a claim to the latest asteroid orbit solution. The source audit found that a newer inspected spacecraft prediction ended before the requested arrival; no extrapolation was used.

For the planetary cache and Mars extraction, load order is LSK, GM, MAR097, DE440s, then the spacecraft SPK where needed. DE440s is loaded after MAR097 so the selected planetary chain is explicit. MAR097 preserves compatibility with the selected mission reconstruction; switching to MAR099 would change the problem. Arrival extraction uses LSK and the merged reference; adding DE440s was independently checked not to change the selected asteroid state. `loaded_kernels` clears both APIs' kernel pools on entry/exit. Use separate processes rather than threaded SPICE sessions.

## Frames, units and time

All reconstructed states use geometric (`NONE`) **J2000** inertial orientation. This is equatorial J2000, not an ecliptic frame. Origins are Earth centre 399, Mars centre 499, or Sun 10 as explicitly labelled. Mars barycentre 4 is not substituted for Mars centre. Tudat states are metres and metres per second; raw CSPICE states are converted from kilometres and kilometres per second. Mass, thrust and time are kg, N and s. ET is represented as TDB seconds since 2000-01-01 12:00:00 TDB.

The pinned constants are:

| Constant | Value |
|---|---:|
| au | 149597870700 m |
| Sun GM | 1.3271244004127939e20 m³/s² |
| Earth GM | 3.986004355070226e14 m³/s² |
| Mars GM | 4.2828373620699086e13 m³/s² |
| Standard gravity for exhaust velocity | 9.80665 m/s² |

Future UTC displays use `naif0012.tls` and may change after future leap-second announcements. The fixed arrival is specified in TDB and is unaffected by that display issue. Numerical digits below describe reproducible kernel evaluations, not measured accuracy.

## Active events and exact stored states

Each state is `[x,y,z,vx,vy,vz]` in SI. Full values, binary segment selection and checks are in the linked JSON files.

**TIP — fixed**, [tip_result.json](artifacts/tip/tip_result.json): launch 2023-10-13 14:19:43 UTC; the post-flight SECO-2 time is launch +3351 s = **15:15:34 UTC**. TIP is +900 s = **2023-10-13 15:30:34 UTC**, ET **750483103.1823578**. The pre-flight schedule gave a time 21 s later and is not used. The source resolves whole seconds and supplies no subsecond uncertainty.

```text
Earth-relative TIP:
[6741926.644469239, 9123110.967067381, 732467.2188004185,
 -1409.0090787061986, 9183.11361019323, 4221.009429301412]
```

The extracted C3 is **34.00229061369918 km²/s²**, versus thesis nominal 34.0 and post-flight rounded 34.0023. Its difference from the nominal is 0.00229061369918 km²/s², inside the thesis ±0.25 requirement. The published whole-second TIP lies inside the launch reconstruction, although SECO-2 itself precedes that SPK's spacecraft coverage. No SECO-2 state is invented.

**Mars SOI entry — fixed**, [mars_entry_result.json](artifacts/soi/mars_entry_result.json): **2026-05-14 07:51:10.173817 UTC**, ET **832017139.3590969**.

```text
Mars-centre-relative entry:
[523275290.6052172, -228950990.3131053, -83483489.39119093,
 -4052.725386430632, 1714.0028059569695, 697.8440470639757]
```

The selected spacecraft segment is Sun-centred at this instant. SPICE translates it to Mars centre using the explicit planetary chain. The radial crossing speed is −4454.604277147173 m/s; the radius residual is approximately 9.48e-5 m. Coarse/fine event scans agree to the stored epoch. Tudat and raw CSPICE reproduce the state exactly after unit conversion; they share CSPICE, so this is an interface consistency check, not independent physical orbit validation.

**Arrival — fixed**, [arrival_result.json](artifacts/arrival/arrival_result.json): **2029-08-01 12:00:00 TDB**, ET **933552000.0**. The LSK-based UTC display is 11:58:50.816739. The spacecraft is constrained to asteroid 2000016's Sun-relative state:

```text
[274723582347.44168, -278206583892.7443, -113373846697.64406,
 12148.871799044442, 13360.252449562, 4709.738180783047]
```

This changes the former noon-UTC boundary by −69.18326056 s and changes its target from spacecraft to asteroid. It is a mathematical rendezvous equality at the user's selected epoch. Asteroid gravity, capture orbit geometry and terminal collision avoidance are outside the retained Sun-only final-phase model.

## SOI convention and event finding

The selected boundary is the fixed-semimajor-axis Laplace sphere:

\[
r_{SOI}=a(\mu_p/\mu_\odot)^{2/5},\qquad
a_E=1\,au,\quad a_M=1.52371034\,au.
\]

The resulting radii are **924646789.2005551 m** for Earth and **577239186.7537631 m** for Mars. These are model choices. Physical gravity does not switch discontinuously at these surfaces.

`soi_events.py` searches the sign of `norm(r_SC/planet) − r_SOI`, brackets crossings, and refines roots with Brent's method and a bisection cross-check. Segment joins are included in the search. Direction is classified using radial crossing rate. Root brackets and floating-point epoch resolution are recorded; displaying microseconds does not imply flight-event knowledge to microseconds.

The retained comparison in [soi/result.json](artifacts/soi/result.json) is:

| Event in reconstructed spacecraft SPK | Fixed-a UTC | Instantaneous heliocentric-distance UTC |
|---|---|---|
| Earth exit | 2023-10-15 09:17:30.504454 | 2023-10-15 09:10:31.405701 |
| Mars entry | 2026-05-14 07:51:10.173817 | 2026-05-14 10:48:56.218372 |
| Mars exit | 2026-05-17 07:04:59.797774 | 2026-05-17 04:09:52.335660 |

Fixed-a events are within about 0.375 s of Earth's SPK origin transition and 1.614 s of the Mars transitions. The instantaneous-distance events differ from those transitions by minutes/hours. This supports the chosen convention's consistency with those files; it does not prove the producer used exactly the same constants. An SPK segment origin transition is a storage/reference-centre change, not a measured physical SOI event.

Only Mars **entry** is fixed to this reconstructed crossing. Earth and Mars **exit** states and epochs are optimisation variables. SPK exits are comparison data. At each optimised exit, the existing equality `norm(r) = r_SOI` and inequality `dot(r,v)/norm(r) >= 0 m/s` apply. This closed nonnegative inequality is the implemented approximation to an outward crossing; a strict mathematical `> 0` cannot be encoded directly as a closed NLP bound. No artificial 1 m/s margin is retained.

## Four-phase model and retained assumptions

| Phase | Origin and gravity | Fixed/free boundaries | Surface constraint |
|---|---|---|---|
| TIP → Earth exit | Earth-centred, Earth point mass only | Fixed TIP; free exit | Earth radius 6378136.6 m |
| Earth exit → Mars entry | Sun-centred, Sun point mass only | Free departure; fixed Mars entry | Sun radius 695700000 m |
| Mars entry → Mars exit | Mars-centred, Mars point mass only | Fixed entry; free exit | Mars radius 3396190 m |
| Mars exit → arrival | Sun-centred, Sun point mass only | Free departure; fixed asteroid rendezvous | Sun radius 695700000 m |

Initial mass is 2747 kg; mass floor is 1662 kg (2747 − thesis xenon mass 1085 kg); maximum thrust is 0.2793 N; constant Isp is 1794 s. Constant thrust capability/Isp, the mass floor and point-mass dynamics are retained modelling assumptions, not a complete operational power/thruster model. No solar radiation pressure, nonspherical gravity, third-body forces, asteroid gravity, eclipses or thrust duty schedule are added.

Earth-centred and Mars-centred durations each have bounds **0.05–10 days**. These are inherited search-domain bounds, not observed mission durations. Solar phase durations follow from those exits and the fixed events and must be positive. All phases retain the same inertial orientation. Origin changes add/subtract the cached planet's position **and velocity** at the variable epoch. Mass is continuous; its initial value alone is fixed. Planetary ephemeris derivatives and phase-duration derivatives enter the exact CasADi derivatives.

There are **no inside/outside-SOI path inequalities**. Surface/mass constraints are evaluated at the existing HS points and at shooting nodes and internal endpoints of the 20 uniform RK7 substeps. Checking these existing endpoints introduces no integration splits. Surface sampling does not certify continuous clearance between samples. There is no separately fixed Mars periapsis or minimum outward speed margin.

## Initialisation, optimisation and scaling

The optimiser is implemented in `deterministic_optimization/` and invoked through the root entry point. Fuel and energy each begin independently with **1e-4 N on each Cartesian thrust component and two uniform RK7 substeps per initial arc**. No spacecraft-SPK history and no previously optimised trajectory seeds these states. The initial mesh is **60/50/90/50**, not the final refined mesh.

The histories in `artifacts/rollout_timing/` are zero-thrust, two-body diagnostic rollouts. Only their Earth/Mars exit durations initialise the two free times. The optimiser regenerates all nodal states with the required small thrust. It propagates TIP → Earth exit → fixed Mars-entry epoch, then resets the second chain's Cartesian initial state to the fixed Mars entry while carrying mass continuously, and propagates Mars exit → arrival. The first chain's Mars-entry mismatch and the final arrival mismatch remain infeasible-seed residuals; no impulse or endpoint snap is inserted.

The optimisation workflow is fresh rollout → same-objective HS once → multiple shooting with local-defect bisection. Controls are constant on each arc:

\[
\mathbf T=T_{max}\alpha\mathbf d,\quad 0\leq\alpha\leq1,\quad
\mathbf d^T\mathbf d=1,\quad \dot m=-T_{max}\alpha/(I_{sp}g_0).
\]

There is no free epigraph mass-flow slack or smoothed norm. Legacy archive keys named `sigma_n` hold the **derived** value `Tmax*alpha`. At zero throttle the direction remains free and may be poorly conditioned. Fuel cost is initial minus final mass, divided by wet mass in the NLP. Energy is **one half** the integral of physical squared thrust, divided by `Tmax² * total transfer duration` in the NLP and reported in N²s. Do not drop the one-half factor when comparing saved objectives.

Planet-centred state scales use length 1e8 m; Sun-centred phases use 1 au. Time is sqrt(length³/GM), velocity is length/time, and mass scale is wet mass. Free exit durations are represented in days. These distinct numerical scales do not change SI reporting.

Multiple shooting uses 20 uniform RK7 substeps per arc. Local defects use 64, with the **optimised throttle** for mass flow and physical thrust vector for acceleration. The error per arc is the maximum of the relative position-vector norm, velocity-vector norm and absolute mass error, each divided by the maximum magnitude at its two endpoints, with a **1e-13 nondimensional denominator floor**. The current target is **5e-14** (the consolidation reference used 1e-13). HS uses cubic Hermite interpolation for the initial shooting handoff; shooting refinement preserves existing nodes and evaluates the preceding shooting arc by partial RK7 propagation at a new node. Controls inherit the parent arc values unchanged.

At most eight refinement passes and 1800 total arcs are allowed. Only NLP convergence and local-defect satisfaction determine acceptance. Failed arcs are bisected; global reintegration never triggers refinement or rejection. Full shared solver settings and commands are in [DETERMINISTIC_OPTIMISATION.md](../DETERMINISTIC_OPTIMISATION.md).

## Planetary cache and numerical limitations

`build_ephemeris_cache.py` follows Tudat's [interpolated SPICE ephemeris API](https://py.api.tudat.space/en/latest/dynamics/environment_setup/ephemeris.html#tudatpy.dynamics.environment_setup.ephemeris.interpolated_spice) and [interpolator API](https://py.api.tudat.space/en/latest/math/interpolators.html). The explicit choice is a natural cubic spline, not Tudat's default Lagrange option. All six state components are fitted independently. The coefficients are stored once and exposed as differentiable CasADi splines; first/second derivatives are checked. No SPICE calls occur inside an objective, constraint, shooting propagation or derivative evaluation.

The accepted grid spacing is **5400 s**, with 24 padding intervals per side. Validity is ET **750483103.1823578–933552069.1832606**. Its upper endpoint retains the earlier noon-UTC cache extent for reproducibility; the actual TDB rendezvous lies inside it. Extrapolation is not an allowed optimisation operation. The numerical interface rejects out-of-range epochs; the symbolic form guards the padded sample domain, and phase-time bounds keep feasible evaluations within validated mission coverage.

The retained validation samples 103757 epochs per body, including knots, off-knot points and reproducible random probes. Maximum errors against the selected SPICE kernels are:

| Body | Position | Velocity | max norm(d fitted position/dt − fitted velocity) |
|---|---:|---:|---:|
| Earth | 0.001463651 m | 4.012702e-9 m/s | 8.548311e-7 m/s |
| Mars | 0.002039433 m | 4.885958e-7 m/s | 1.089895e-6 m/s |

Engineering cache thresholds are 0.1 m, 1e-5 m/s and 1e-4 m/s respectively. These sampled interpolation errors do not bound the physical ephemeris error. Native Tudat and CasADi backend checks are stored in [validation.json](artifacts/ephemerides/validation.json). Position and velocity fits are not exactly kinematically identical; use the actual fitted-position derivative when differentiating a moving-origin constraint. Rebuilding provenance changes archive bytes/hashes even when event states and tabulated ephemerides agree; the pinned cache is retained for exact batch comparisons.

Continuous propagation uses DOP853 as a **report-only** diagnostic, both with the throttle mass-flow law and with the norm of actual physical thrust. The tighter reference comparison carries the state through all phases, translating origins without resetting it to optimised boundary states. Its tolerance studies are numerical sensitivity checks. In the reference fuel and energy solutions, final position discrepancies were approximately **696 m** and **145 m**, while tighter-versus-tightest reference differences were approximately **1033 m** and **1089 m**. Consequently, those apparent endpoint errors do not establish sub-kilometre global accuracy. Local defects near 1e-13 are not a bound on multi-year accumulated error, ephemeris uncertainty or physical mission fidelity.

The reconstructed SPK comparisons in `artifacts/soi/` concern event extraction and two-body timing rollouts. They are kept separate from optimiser feasibility and from the final asteroid rendezvous assumption. The available reconstructed SPKs do not cover the full transfer; no missing flight trajectory is fabricated.

## Files

* `tip.py`, `kernels.py`: post-flight timing, launch state, C3, coverage and kernel management.
* `soi_events.py`, `mission_soi.py`, `soi_reconstruction.py`, `mars_kernels.py`: crossing definitions, root refinement, fixed Mars entry, source comparisons.
* `reference_kernel.py`, `reconstruct_arrival.py`, `arrival_boundary.py`: predicted asteroid state and strict fixed-boundary loader.
* `ephemeris_cache.py`, `build_ephemeris_cache.py`: stored tables, differentiable interpolation, validation.
* `orbit.py`, `rollout.py`, `seed_earth_departure.py`, `seed_mars_flyby.py`, `seed_solar_phases.py`: independently checked two-body timing rollouts.
* `paths.py`, `verify.py`, `__main__.py`, `download_kernels.py`: input selection, provenance checks and reproduction commands.
* `artifacts/`: pinned TIP, SOI, asteroid, ephemeris and timing products. `evidence/` contains clearly labelled historical source audits, not active alternative mission boundaries.
