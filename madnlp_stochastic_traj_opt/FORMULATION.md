# Problem formulation

Closed-loop stochastic low-thrust transfer with two problems on the same model:

- **Step 3** (`--objective=standard`, the default): a tracking cost (mean thrust, state and control dispersion,
  terminal dispersion), subject to the moment dynamics and a thrust chance constraint per arc.
- **Step 4** (`--objective=chance_sum`): minimise the time-weighted sum of the per-arc thrust chance bounds, subject
  to the moment dynamics, the thrust chance constraints and a hard bound on the terminal covariance.

## Variables (arc k = 1..N, node k = 1..N+1)

| Symbol | Meaning |
|---|---|
| $m_k \in \mathbb{R}^7$ | mean state (position, velocity, mass) at node k |
| $\Gamma_k \in [0,1]$, $d_k$ with $\lVert d_k\rVert^2 = 1$ | throttle and unit direction: feedforward thrust $S_k = \Gamma_k d_k$ (in $T_{max}$) |
| $\hat K_k \in \mathbb{R}^{3\times 6}$ | scaled feedback gain on the position/velocity deviation |
| $L_k$ (lower 7×7) | scaled Cholesky factor of the state covariance at node k |
| $\hat\tau_k \ge 0$, $\hat G_k$ (lower 3×3) | step 4: epigraph of the executed-control spread (below) |
| $M$ (lower 6×6) | step 4: terminal margin factor |

Policy on arc k: $u = \Gamma_k d_k + K_k(\hat x - m_k)$, with $\hat x$ the navigated state (true state plus
navigation error), executed with the proportional Gates error (magnitude and pointing). On Psyche's Mars phase
there is no control.

## Moment propagation

A nested unscented transform per arc: 32 outer sigma points (16 augmented dimensions — state, navigation, process
noise — with $\kappa = 0$, so there is no centre point), each with 6 Gates execution points, propagated by one RK9
step. Defect rows:

- mean: $m_{k+1} - \bar y_k = 0$
- covariance, square-root form: $L_{k+1} - \operatorname{chol}(V_k) = 0$, with $V_k$ the scaled UT covariance

The sigma-point mass flow is $\sqrt{\lVert u\rVert^2 + (\delta\,T_{max})^2}$ with $\delta = 5\cdot10^{-4}$ on the 24
outer points whose command carries feedback, and the exact $\lVert u\rVert$ elsewhere. $\Gamma$ and the Gates factors
are exact.

## Thrust chance constraint (per powered arc)

$P(\lVert u_{exec}\rVert > T_{max}) \le \beta$ with $\beta = 0.01$, imposed as

$$\Gamma_k + \psi\,\rho(T_k) \le 1,\qquad \rho(T_k) = \sqrt{\lambda_{max}(T_k)},\qquad \psi = \sqrt{\chi^2_{3}{}^{-1}(1-\beta)} \approx 3.37,$$

where $T_k$ is the executed-control covariance in closed form: feedback dispersion (state and navigation) plus the
Gates execution covariance.

- **Step 3** evaluates $\rho$ with the trigonometric 3×3 eigenvalue formula, smoothed by $s = 10^{-6}$ in
  $p = \sqrt{p_2/6 + s^2}$, and $\rho = \sqrt{\lambda_{max} + f^2}$ with $f = 10^{-6}$.
- **Step 4** replaces $\rho$ by an epigraph: $\tau_k = s_k\hat\tau_k \ge \rho(T_k)$ through
  $$\hat\tau_k^2 I - \hat G_k \hat G_k^\top - T_k / s_k^2 = 0, \qquad \operatorname{diag}(\hat G_k) \ge f_G / s_k,$$
  with $s_k$ the value of $\rho(T_k)$ at the start (a fixed scale) and $f_G = 5\cdot10^{-4}$. All six rows are smooth
  (no eigenvalue is computed), and the chance row becomes $\Gamma_k + \psi s_k \hat\tau_k \le 1$. The floor bounds
  the conservatism: $\tau_k - \rho_k \le \sqrt{\rho_k^2 + f_G^2} - \rho_k$.

## Objective (arc weights $w_k = N\,\Delta t_k / (t_f - t_0)$)

Step 3, with the tracking weights $w_P$, $w$ of the case and $c = 3$:

$$J_3 = \sum_{k\ \text{powered}} w_k \left(\Gamma_k + w\,\operatorname{tr} T_k\right) + \frac{w_P}{c^2} \sum_k w_k \operatorname{tr} P_k[1{:}6,1{:}6] + \frac{1}{c^2}\operatorname{tr}\left(G P_{N+1} G^\top\right),$$

$G = \operatorname{diag}(1/\sqrt{r})$ the terminal target (the requested position/velocity variance reductions $r$;
Psyche: 1500 km and 2 m/s per axis).

Step 4: $J_4 = \sum_k w_k \left(\Gamma_k + \psi\, s_k \hat\tau_k\right)$, the time-weighted sum of the chance-row
left-hand sides, i.e. of the per-arc 99 % bound on the executed thrust magnitude. The cost pushes every $\tau_k$ down
onto $\rho(T_k)$ (up to the floor), so at the optimum $J_4$ equals $\sum_k w_k(\Gamma_k + \psi\rho(T_k))$.

## Boundary and terminal conditions

- $m_1 = x_0$; mean position and velocity at the target at node $N+1$.
- Step 4: hard terminal bound $G P_{N+1} G^\top \preceq I$, written $I - G P_{N+1} G^\top - M M^\top = 0$,
  $\operatorname{diag}(M) \ge 10^{-5}$.
- Psyche adds the Mars B-plane clearance chance row (500 km at risk $10^{-3}$); its ballistic Mars arcs have no
  control.

## Initial guess and scaling

- Fuel-optimal deterministic reference, re-optimised for fuel on the stochastic mesh (throttle and direction):
  means, $\Gamma$, $d$. Single-phase cases: the reference's own mesh with every interval split in two; Psyche: the
  exported fuel-optimal mission on its mesh refined ×2.
- TVLQR gains with $\tilde Q = w_P I/c^2$, $\tilde R = w I$, $c = 3$, weighted per arc by $w_k$: the cost's weights in
  step 3, Bryson's rule $w_P = w = 1$ in step 4 (Psyche: the cost's weights in both); covariances propagated by the UT.
- Covariance coordinates $D_0^{-1}\Sigma D_0^{-1}$ (initial standard deviations); factor variables scaled by the seed
  node standard deviations (Psyche: full lower scales $S_k = \operatorname{chol}(P_k^{seed})$).

## Solver

MadNLP interior point with exact Hessians, sparse KKT system with MUMPS and exact equalities, tol $10^{-6}$, monotone
barrier from $\mu_0 = 0.1$, `constr_mult_init_max` $10^4$, at most 5000 (step 3) or 10 000 (step 4) iterations. The
model is evaluated on the CPU (`--mode=cpu`) or on a CUDA GPU (`--mode=gpu`); MadNLP and MUMPS run on the CPU in both.
