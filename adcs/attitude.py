"""
Attitude representations: quaternions and rotation (attitude) matrices.

Conventions (fixed here, used everywhere else in this project):

- Quaternion is scalar-last: q = [q1, q2, q3, q4], with q4 the scalar part and
  [q1, q2, q3] the vector part. Unit norm.

- The attitude matrix A(q) rotates INERTIAL-frame vector components into
  BODY-frame components: v_B = A(q) @ v_I. This matches how sensors measure
  (in the body frame) against environment models that provide truth vectors
  (in the inertial frame)

- Formula taken from "Fundamentals of Spacecraft Attitude Determination and Control" 
  chapter 2.

"""
import numpy as np
from dataclasses import dataclass

def quat_to_dcm(q):
    """Convert a quaternion to a direction cosine matrix according to Eq. 2.125"""
    q1, q2, q3, q4 = q
    return np.array([
        [q1**2 + q4**2 - q2**2 - q3**2, 2*(q1*q2 + q3*q4), 2*(q1*q3 - q2*q4)],
        [2*(q1*q2 - q3*q4), q2**2 + q4**2 - q1**2 - q3**2, 2*(q2*q3 + q1*q4)],
        [2*(q1*q3 + q2*q4), 2*(q2*q3 - q1*q4), q3**2 + q4**2 - q1**2 - q2**2]
    ])

def dcm_to_quat(A):
    """Convert a direction cosine matrix to a quaternion according to Eq. 2.135"""
    p12 = A[0,1] + A[1,0]
    p13 = A[0,2] + A[2,0]
    p23 = A[1,2] + A[2,1]
    m23 = A[1,2] - A[2,1]
    m31 = A[2,0] - A[0,2]
    m12 = A[0,1] - A[1,0]

    traceA = np.trace(A)
    index = np.argmax([A[0,0], A[1,1], A[2,2], traceA])

    if index == 0:
        q = [1+ 2* A[0,0]- traceA, p12, p13, m23]
        q /= 2*np.sqrt(q[0])
        return q
    elif index == 1:
        q = [p12, 1+ 2* A[1,1]- traceA, p23, m31]
        q /= 2*np.sqrt(q[1])
        return q
    elif index == 2:
        q = [p13, p23, 1+ 2* A[2,2]- traceA, m12]
        q /= 2*np.sqrt(q[2])
        return q
    else:
        q = [m23, m31, m12, 1+ traceA]
        q /= 2*np.sqrt(q[3])
        return q

def quat_multiply(q1, q2):
    """Quaternion Multiplication according to Eq. 2.82a."""
    q1_vec = q1[:3]
    q1_scalar = q1[3]
    q2_vec = q2[:3]
    q2_scalar = q2[3]

    return np.append(q1_scalar*q2_vec + q2_scalar*q1_vec - np.cross(q1_vec, q2_vec),
        q1_scalar*q2_scalar - np.dot(q1_vec, q2_vec))

def rotate_vector(q, v_I):
    """Rotate a vector v_I from inertial frame to body frame using quaternion q."""
    A = quat_to_dcm(q)
    return A @ v_I

def quat_from_axis_angle(axis, angle):
    """Convert an axis-angle representation to a quaternion."""
    axis = axis / np.linalg.norm(axis)  # Ensure the axis is a unit vector
    half_angle = angle / 2.0
    q_scalar = np.cos(half_angle)
    q_vector = axis * np.sin(half_angle)
    return np.append(q_vector, q_scalar)

@dataclass
class State:
    omega: np.ndarray
    q: np.ndarray

    def __add__(self, val2):
        return State(self.omega + val2.omega, self.q + val2.q)
    
    def __mul__(self, scalar):
        return State(self.omega * scalar, self.q * scalar)
    
    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def normalize_q(self):
        return State(self.omega, self.q / np.linalg.norm(self.q))