import numpy as np

from adcs.attitude import quat_multiply

def dynamics(state, torque, J):
    J_inv = np.linalg.inv(J)
    omega, q = state
    omega_dot = J_inv @ (torque - np.cross(omega, J @ omega))
    q_dot = 0.5 * quat_multiply(np.append(omega, 0), q)

    return (omega_dot, q_dot)
