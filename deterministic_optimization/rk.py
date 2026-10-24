"""The original RK7 tableau and arithmetic for numerical rollout/defects."""

from integrator import RK7_A, RK7_B
import numpy as np


def rk7(rhs, state, dt):
    stages = []
    for i in range(len(RK7_B)):
        value = state.copy()
        for j in range(i):
            if RK7_A[i, j]:
                value += dt * RK7_A[i, j] * stages[j]
        stages.append(rhs(value))
    return state + dt * np.einsum("i,ij->j", RK7_B, np.array(stages))
