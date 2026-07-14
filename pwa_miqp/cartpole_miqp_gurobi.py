"""
Cart-pole swing-up MIQP with PWA dynamics — Python + Gurobi.

State (4-dim):
    x[0] = cart position       (x_c)
    x[1] = pole angle θ        (θ = 0 → pole hanging DOWN, θ = π → UP)
    x[2] = cart velocity       (ẋ_c)
    x[3] = pole angular vel    (θ̇)
Control: u = horizontal force on the cart.

PWA decomposition: 4 modes on the pole-angle circle, centred at
    θ ∈ {0,  π/2,  π,  3π/2}
each spanning π/2; this matches pendulum_miqp_gurobi.py exactly so that the
two scripts share the same mode-adjacency / big-M / indicator structure.

Boundary conditions:
    x_start = (0, 0, 0, 0)    cart at origin, pole down, at rest
    x_goal  = (0, π, 0, 0)    cart at origin, pole upright, at rest

Same Gurobi-native features as the pendulum file:
    - MIQP quadratic objective ½ dt Σ u²
    - General-constraint INDICATORS for both polytope and dynamics
    - SOS1 on z[:, t]
    - Mode-adjacency cuts on the angle circle
    - 30-minute TimeLimit, 1% MIPGap (to allow proving optimality if it can)
"""
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import gurobipy as gp
from gurobipy import GRB
import time

jax.config.update("jax_enable_x64", True)


# ---- 1. Continuous-time cart-pole dynamics -------------------------------
#   Lagrangian for θ measured from straight-down (θ = 0 is pole down):
#     (m_c + m_p) ẍ_c + m_p L cos(θ) θ̈ − m_p L sin(θ) θ̇²  =  u
#     m_p L cos(θ) ẍ_c + m_p L² θ̈ + m_p g L sin(θ)        =  0
M_CART  = 1.0      # cart mass [kg]
M_POLE  = 0.1      # pole mass [kg]
L_POLE  = 0.5      # pole length to COM [m]
GRAV    = 9.81     # gravity [m/s²]

def f_cont(x, u):
    """Continuous-time RHS, returns ẋ = [ẋ_c, θ̇, ẍ_c, θ̈]"""
    th, x_dot, th_dot = x[1], x[2], x[3]
    # Mass matrix entries
    m11 = M_CART + M_POLE
    m12 = M_POLE * L_POLE * jnp.cos(th)
    m22 = M_POLE * L_POLE ** 2
    det = m11 * m22 - m12 ** 2          # = m_p L² (m_c + m_p sin²(θ))
    # RHS of M [ẍ_c; θ̈] = rhs
    rhs1 = u + M_POLE * L_POLE * jnp.sin(th) * th_dot ** 2
    rhs2 = -M_POLE * GRAV * L_POLE * jnp.sin(th)
    # M^-1 · rhs
    x_ddot  = ( m22 * rhs1 - m12 * rhs2) / det
    th_ddot = (-m12 * rhs1 + m11 * rhs2) / det
    return jnp.stack([x_dot, th_dot, x_ddot, th_ddot])


df_dx = jax.jit(jax.jacfwd(f_cont, argnums=0))
df_du = jax.jit(jax.jacfwd(f_cont, argnums=1))


def linearise_matexp(x_eq, u_eq, dt):
    """Discrete x_next = A_d x + B_d u + c_d via matrix exponential
    on the augmented affine system."""
    x_eq_j = jnp.asarray(x_eq, dtype=jnp.float64)
    u_eq_j = jnp.float64(u_eq)
    A_c = np.asarray(df_dx(x_eq_j, u_eq_j))
    B_c = np.asarray(df_du(x_eq_j, u_eq_j)).reshape(-1, 1)
    f_eq = np.asarray(f_cont(x_eq_j, u_eq_j))
    d_abs = f_eq - A_c @ x_eq - B_c.flatten() * u_eq

    n, m = x_eq.size, 1
    M = np.zeros((n + m + n, n + m + n))
    M[:n, :n]    = A_c
    M[:n, n:n+m] = B_c
    M[:n, n+m:]  = np.eye(n)
    M_d = scipy.linalg.expm(M * dt)
    A_d = M_d[:n, :n]
    B_d = M_d[:n, n:n+m].flatten()
    D_d = M_d[:n, n+m:]
    c_d = D_d @ d_abs
    return A_d, B_d, c_d


# ---- 2. Problem data -------------------------------------------------------
DT          = 0.1
T           = 50                        # 5-second horizon
U_MAX       = 20.0                      # max horizontal force on cart [N]
X_CART_MAX  = 2.0                       # cart position bound [m]
V_CART_MAX  = 10.0                      # cart velocity bound [m/s]
OMEGA_MAX   = 10.0                      # pole angular velocity bound [rad/s]
NUM_MODES   = 4

theta_centres = [0.0, np.pi/2, np.pi, 3*np.pi/2]
theta_bounds  = [(-np.pi/4,    np.pi/4),
                 ( np.pi/4,  3*np.pi/4),
                 (3*np.pi/4,  5*np.pi/4),
                 (5*np.pi/4,  7*np.pi/4)]

I_START, I_GOAL = 0, 2
x0_state = np.array([0.0, 0.0, 0.0, 0.0])     # cart at origin, pole DOWN
xf_state = np.array([0.0, np.pi, 0.0, 0.0])   # cart at origin, pole UP

# Per-mode discrete affine dynamics, linearised at the pole-angle centre with
# everything else at zero (cart at origin, all velocities zero).
A_m, B_m, c_m = {}, {}, {}
for i in range(NUM_MODES):
    x_eq = np.array([0.0, theta_centres[i], 0.0, 0.0])
    A_m[i], B_m[i], c_m[i] = linearise_matexp(x_eq, 0.0, DT)
    print(f"mode {i}  θ_eq={theta_centres[i]:+.3f}: "
          f"‖c‖∞ = {np.max(np.abs(c_m[i])):.4f}")

# Circle-adjacency on modes
adjacent = set()
for i in range(NUM_MODES):
    for j in range(NUM_MODES):
        if (i == j) or (abs(i - j) == 1) or (abs(i - j) == NUM_MODES - 1):
            adjacent.add((i, j))


# ---- 3. Build the Gurobi MIQP ----------------------------------------------
m = gp.Model("cartpole_pwa_miqp")
m.setParam("OutputFlag", 1)
m.setParam("MIPGap", 0.01)
m.setParam("TimeLimit", float(__import__("os").environ.get("MIQP_TIME_LIMIT", 1800.0)))  # 30-min default; override via env
m.setParam("Threads", 0)

# ---- LB-acceleration parameters ----
# MIPFocus=3 tells Gurobi to prioritise proving the lower bound (since we
# already get a good incumbent fast).  Cuts=3 + aggressive presolve push the
# QP root higher.  Disabling heuristics keeps Gurobi from spending time
# searching for better incumbents we don't need.
m.setParam("MIPFocus",     3)
m.setParam("Cuts",         3)
m.setParam("Heuristics",   0)
m.setParam("RINS",         0)
m.setParam("Method",       1)   
m.setParam("Presolve",     2)
m.setParam("PreSparsify",  1)

# --- 3a. Variables ---
#variable bounds
#divided by 4 because we have 4 regions, these are necessary 
TH_W = 2 * np.pi / 4 
TH_LO = -TH_W / 2
TH_HI = 2 * np.pi - TH_W / 2

#statelower and upper bounds
state_lb = np.array([[-X_CART_MAX] * (T + 1), [TH_LO] * (T + 1),
                     [-V_CART_MAX] * (T + 1), [-OMEGA_MAX] * (T + 1)])
state_ub = np.array([[ X_CART_MAX] * (T + 1), [TH_HI] * (T + 1),
                     [ V_CART_MAX] * (T + 1), [ OMEGA_MAX] * (T + 1)])



x = m.addMVar((4, T + 1), lb=state_lb, ub=state_ub, name="x")
u = m.addMVar(T, lb=-U_MAX, ub=U_MAX, name="u")
z = m.addMVar((NUM_MODES, T), vtype=GRB.BINARY, name="z")

# --- 3b. Boundary conditions ---
for k in range(4):
    m.addConstr(x[k, 0] == x0_state[k], name=f"x0_{k}")
    m.addConstr(x[k, T] == xf_state[k], name=f"xf_{k}")

# --- 3c. Mode exclusivity (sum=1) + SOS1 (special ordered set constraint)---
for t in range(T):
    #sum of the binaries at each timestep = 1
    m.addConstr(z[:, t].sum() == 1, name=f"sum_z_{t}")
    #at most one of the variables is nonzero
    m.addSOS(GRB.SOS_TYPE1, [z[i, t].item() for i in range(NUM_MODES)])

# --- 3d. Mode polytope via INDICATORS (only θ partitioned) ---
#add the bounds dependent on which mode is active using indicator constraints
for i in range(NUM_MODES):
    th_lo, th_hi = theta_bounds[i]
    for t in range(T):
        m.addGenConstrIndicator(z[i, t].item(), True,
                                x[1, t].item(), GRB.GREATER_EQUAL, th_lo,
                                name=f"poly_lo_{i}_{t}")
        m.addGenConstrIndicator(z[i, t].item(), True,
                                x[1, t].item(), GRB.LESS_EQUAL, th_hi,
                                name=f"poly_hi_{i}_{t}")

# --- 3e. PWA dynamics via INDICATORS ---
#   z[i, t] = 1  ⇒  x[k, t+1] = Σ_j A_i[k, j] x[j, t] + B_i[k] u[t] + c_i[k]
for i in range(NUM_MODES):
    A_i, B_i, c_i = A_m[i], B_m[i], c_m[i]
    for t in range(T):
        for k in range(4):
            lhs = (x[k, t + 1]
                   - A_i[k, 0] * x[0, t] - A_i[k, 1] * x[1, t]
                   - A_i[k, 2] * x[2, t] - A_i[k, 3] * x[3, t]
                   - B_i[k]    * u[t])
            m.addGenConstrIndicator(z[i, t].item(), True,
                                    lhs.item(), GRB.EQUAL, c_i[k],
                                    name=f"dyn_{i}_{t}_{k}")

# --- 3f. Mode-adjacency cuts ---
for t in range(T - 1):
    for i in range(NUM_MODES):
        for j in range(NUM_MODES):
            if (i, j) not in adjacent:
                m.addConstr(z[i, t] + z[j, t + 1] <= 1,
                            name=f"adj_{i}_{j}_{t}")

# --- 3g. Symmetry break: start in mode 0 (which matches x0_state) ---
m.addConstr(z[I_START, 0] == 1, name="start_mode")

# --- 3g.1. Mode-reachability valid inequalities ---
# Adjacency forces ≥ 2 mode transitions to get from mode 0 (start) to mode 2
# (goal). So mode 2 cannot be active in the first 2 timesteps, and mode 0
# cannot be active in the last 2.  These cuts are redundant for any feasible
# trajectory but kill the corresponding LP relaxation branches up front.
m.addConstr(z[I_GOAL,  0] == 0, name="reach_goal_t0")
m.addConstr(z[I_GOAL,  1] == 0, name="reach_goal_t1")
m.addConstr(z[I_START, T - 1] == 0, name="reach_start_tT-1")
m.addConstr(z[I_START, T - 2] == 0, name="reach_start_tT-2")

# --- 3h. Quadratic objective ½ dt Σ u² ---
m.setObjective(0.5 * DT * (u @ u), GRB.MINIMIZE)

m.update()
print(f"\nModel: {m.NumVars} vars ({m.NumIntVars} int), "
      f"{m.NumConstrs} linear cons, {m.NumGenConstrs} indicator cons, "
      f"{m.NumSOS} SOS")


# ---- 4. Solve -------------------------------------------------------------
t0 = time.time()
m.optimize()
elapsed = time.time() - t0
print(f"\nSolve took {elapsed:.2f} s, status = {m.Status}")
print(f"Objective : {m.ObjVal:.6f}")
print(f"MIPGap    : {m.MIPGap:.4f}")
print(f"NodeCount : {m.NodeCount}")


# ---- 5. Extract and plot ---------------------------------------------------
x_val = x.X
u_val = u.X 
z_val = z.X
mode_seq = np.argmax(z_val, axis=0)
dwell = {i: int(np.sum(mode_seq == i)) for i in range(NUM_MODES)}
print(f"Active mode sequence (0=down, 1=π/2, 2=up, 3=3π/2):")
print(mode_seq.tolist())
print(f"Dwell per mode: {dwell}")

ts = np.arange(T + 1) * DT
fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

axes[0].plot(ts, x_val[0], 'b-o', ms=3, lw=1.6)
axes[0].axhline(0, ls=':', c='k'); axes[0].set_ylabel(r'$x_c$ [m]')
axes[0].grid(alpha=.3)

axes[1].plot(ts, x_val[1], 'g-o', ms=3, lw=1.6)
axes[1].axhline(np.pi, ls=':', c='g', label=r'goal $\pi$')
axes[1].axhline(0, ls=':', c='k', label='start')
axes[1].set_ylabel(r'$\theta$ [rad]'); axes[1].grid(alpha=.3); axes[1].legend()

axes[2].plot(ts, x_val[2], 'c-o', ms=3, lw=1.6)
axes[2].set_ylabel(r'$\dot{x}_c$ [m/s]'); axes[2].grid(alpha=.3)

axes[3].plot(ts, x_val[3], 'r-o', ms=3, lw=1.6)
axes[3].set_ylabel(r'$\dot{\theta}$ [rad/s]'); axes[3].grid(alpha=.3)

axes[4].step(ts[:-1], u_val, where='post', color='k', lw=1.6)
axes[4].axhline( U_MAX, ls=':', c='r'); axes[4].axhline(-U_MAX, ls=':', c='r')
axes[4].set_ylabel(r'$u$ [N]'); axes[4].grid(alpha=.3)

axes[5].step(ts[:-1], mode_seq, where='post', color='purple', lw=1.6)
axes[5].set_yticks(range(NUM_MODES))
axes[5].set_yticklabels(['0 (down)', 'π/2', 'π (up)', '3π/2'])
axes[5].set_ylabel('mode'); axes[5].set_xlabel('t [s]'); axes[5].grid(alpha=.3)

fig.suptitle(f'Cart-pole MIQP (Gurobi), cost={m.ObjVal:.3f}, '
             f'solve={elapsed:.1f}s, gap={m.MIPGap*100:.2f}%, modes={dwell}')
plt.tight_layout()
plt.savefig('cartpole_miqp_gurobi.png', dpi=150)
print("Saved plot → cartpole_miqp_gurobi.png")
