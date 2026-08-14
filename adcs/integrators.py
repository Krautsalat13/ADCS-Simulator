import numpy as np
from adcs.dynamics import dynamics

def make_dynamics_rhs(torque, J):
    def rhs_fn(state):
        return dynamics(state, torque, J)
    return rhs_fn



def rk4_step(state, rhs_fn, dt):
    k1 = rhs_fn(state) * dt
    k2 = rhs_fn(state + 0.5 * k1) * dt
    k3 = rhs_fn(state + 0.5 * k2) * dt
    k4 = rhs_fn(state + k3) * dt
    state += (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    return state.normalize_q()

