# Hybrid KKT regularization support

`regularized_hybrid_kkt.jl` wraps `HybridKKT.HybridCondensedKKTSystem`. It uses
the package's sparse GPU matrices, cuDSS factorization, and conjugate-gradient
workspace. It adds support for the dual diagonal that the pinned HybridKKT
version omits in its condensed solve. That diagonal is used by MadNLP's
inertia correction and robust restoration.

The adapter changes the linear algebra of a regularized Newton step. The
objective, constraints, UT propagation, and number of optimization variables
are those of the original transcription.

## Condensation

After eliminating the bound multipliers, let `Σs` be the slack diagonal,
`Di ≤ 0` the inequality dual diagonal, and `De ≤ 0` the equality dual diagonal.
The inequality condensation uses

```
T = inv(I - Di*Σs)
D = Σs*T
Hbar = H + Σx + A'*D*A
```

Here `A` is the inequality Jacobian and `G` is the equality Jacobian. The
remaining equations are

```
Hbar*dx + G'*dy = bx
G*dx + De*dy = be
```

Choose an effective augmentation `γ` no larger than `hybrid_gamma`, with
`γ*maximum(abs, De) ≤ 1/2`. This makes `B = I + γ*De` positive. Factor

```
Kγ = Hbar + γ*G'*G
```

and solve the symmetric Schur system for `v = B*dy`:

```
(G*inv(Kγ)*G' - De*inv(B))*v = G*inv(Kγ)*(bx + γ*G'*be) - be
```

Then recover `dx`, `dy`, the slack step, and the bound-multiplier steps. When
the dual diagonal is zero, these equations reduce to the package's original
hybrid algorithm. MadNLP's standard residual-based iterative refinement is
used with the adapter.

## Verification

`hybrid_kkt_tests.jl` checks the full uncondensed residual on a GPU test problem
with variable bounds, equality constraints, and both one-sided and two-sided
inequalities. It exercises zero dual diagonals, equality-only regularization,
inequality-only regularization, and simultaneous nonuniform regularization.
All eight checks passed before the full trajectory solve was attempted.

Run these checks together with the complete CPU/GPU derivative comparisons:

```sh
julia --project=madnlp_stochastic_traj_opt madnlp_stochastic_traj_opt/test/runtests_gpu.jl --derivatives-only
```

The relevant upstream implementation is pinned in
[HybridKKT's KKT source](https://github.com/madsuite-org/HybridKKT.jl/blob/fc1751c1f3b2f41afaa7168659270c2d13168d79/src/kkt.jl).
