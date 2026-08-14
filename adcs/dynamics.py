import numpy as np

from adcs.attitude import quat_multiply, State

    
def dynamics(state, torque, J):
    J_inv = np.linalg.inv(J)
    omega_dot = J_inv @ (torque - np.cross(state.omega, J @ state.omega))
    q_dot = 0.5 * quat_multiply(np.append(state.omega, 0), state.q)
    return State(omega_dot, q_dot)

