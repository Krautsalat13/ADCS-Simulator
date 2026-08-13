import numpy as np
import pytest

from adcs.attitude import (
    quat_to_dcm,
    dcm_to_quat,
    quat_multiply,
    rotate_vector,
    quat_from_axis_angle,
)


def _random_unit_quaternions(n, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def _closest_sign(q_recovered, q):
    # q and -q represent the same attitude, so pick the sign that matches
    # before comparing componentwise.
    if np.dot(q_recovered, q) < 0:
        return -q_recovered
    return q_recovered


def test_quat_dcm_round_trip():
    for q in _random_unit_quaternions(200, seed=0):
        A = quat_to_dcm(q)
        q_recovered = _closest_sign(np.asarray(dcm_to_quat(A)), q)
        assert np.allclose(q_recovered, q, atol=1e-12)


@pytest.mark.parametrize("q_raw", [
    [0.9, 0.1, 0.1, None],   # forces the q1-dominant pivot branch
    [0.1, 0.9, 0.1, None],   # forces the q2-dominant pivot branch
    [0.1, 0.1, 0.9, None],   # forces the q3-dominant pivot branch
    [0.05, 0.05, 0.05, None],  # forces the q4 (trace)-dominant pivot branch
])
def test_dcm_to_quat_pivot_branches(q_raw):
    q1, q2, q3, _ = q_raw
    q4 = np.sqrt(1 - q1**2 - q2**2 - q3**2)
    q = np.array([q1, q2, q3, q4])
    A = quat_to_dcm(q)
    q_recovered = _closest_sign(np.asarray(dcm_to_quat(A)), q)
    assert np.allclose(q_recovered, q, atol=1e-12)


def test_quat_multiply_matches_dcm_composition():
    q1s = _random_unit_quaternions(100, seed=1)
    q2s = _random_unit_quaternions(100, seed=2)
    for q1, q2 in zip(q1s, q2s):
        lhs = quat_to_dcm(quat_multiply(q1, q2))
        rhs = quat_to_dcm(q1) @ quat_to_dcm(q2)
        assert np.allclose(lhs, rhs, atol=1e-12)


def test_rotate_vector_known_axis_angle():
    # 90 deg about z applied to the inertial x-axis: body frame is rotated
    # +90 deg about z relative to inertial, so inertial x appears along
    # body -y (see the worked example in attitude.py's development notes).
    q = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    v_B = rotate_vector(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(v_B, [0.0, -1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("axis,angle", [
    ([1.0, 0.0, 0.0], 0.3),
    ([0.0, 1.0, 0.0], 2.1),
    ([1.0, 1.0, 1.0], -1.2),
])
def test_quat_from_axis_angle_is_unit_norm(axis, angle):
    q = quat_from_axis_angle(np.array(axis), angle)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)
