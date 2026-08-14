import numpy as np
import pytest
from pathlib import Path

from adcs.dynamics import dynamics
from adcs.config import load_mission_config
from adcs.attitude import State
from adcs.integrators import make_dynamics_rhs, rk4_step

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "mission.yaml"
mission_config = load_mission_config(CONFIG_PATH)

def test_rk4_step_principal_axis():
    # propagating a principal-axis spin under zero torque should hold the rate
    # constant over many steps and keep the quaternion normalized throughout
    J = mission_config.spacecraft.inertia_tensor_kgm2
    J_EV = np.linalg.eigh(J)

    torque = np.array([0.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.0, 1.0])
    dynamics_rhs = make_dynamics_rhs(torque, J)
    dt = 0.01

    for i in range(3):
        omega = J_EV.eigenvectors[:, i]
        omega = omega / np.linalg.norm(omega)
        state = State(omega, q)

        t = 0
        while t < 5:
            state = rk4_step(state, dynamics_rhs, dt)
            t += dt
            assert np.allclose(dynamics(state, torque, J).omega, 0, atol=1e-12)
            assert np.allclose(np.linalg.norm(state.q), 1, atol=1e-12)
