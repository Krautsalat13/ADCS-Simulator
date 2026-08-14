import numpy as np
import pytest
from pathlib import Path

from adcs.dynamics import dynamics
from adcs.config import load_mission_config
from adcs.attitude import State

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "mission.yaml"
mission_config = load_mission_config(CONFIG_PATH)

def test_dynamics():
    # a rate aligned with a principal axis is an equilibrium of Euler's equations
    J = mission_config.spacecraft.inertia_tensor_kgm2
    J_EV = np.linalg.eigh(J)

    torque = np.array([0.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.0, 1.0])

    for i in range(3):
        omega = J_EV.eigenvectors[:, i]
        omega = omega / np.linalg.norm(omega)
        state = State(omega, q)

        assert np.allclose(dynamics(state, torque, J).omega, 0, atol=1e-12)
