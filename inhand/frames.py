"""Palm-frame transforms, the symmetric goal encoding, and the error metrics."""

from __future__ import annotations

import mujoco
import numpy as np


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return q


def rotvec_to_mat(v: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, v / max(np.linalg.norm(v), 1e-12), np.linalg.norm(v))
    return quat_to_mat(q)


def in_palm(p_palm: np.ndarray, R_palm: np.ndarray, p: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World pose -> (position, cylinder axis) in the palm frame."""
    return R_palm.T @ (p - p_palm), R_palm.T @ R[:, 2]


def axis_outer(a: np.ndarray) -> np.ndarray:
    """Sign-invariant axis encoding: upper triangle of a a^T. Same for a and -a."""
    a = a / max(np.linalg.norm(a), 1e-9)
    return np.array([a[0] * a[0], a[1] * a[1], a[2] * a[2], a[0] * a[1], a[0] * a[2], a[1] * a[2]])


def errors(p: np.ndarray, a: np.ndarray, p_t: np.ndarray, a_t: np.ndarray) -> tuple[float, float]:
    """(position error, axis tilt). Roll and end-flip are ignored, per the task."""
    c = abs(float(np.dot(a, a_t))) / max(np.linalg.norm(a) * np.linalg.norm(a_t), 1e-9)
    return float(np.linalg.norm(p - p_t)), float(np.arccos(np.clip(c, 0, 1)))


def random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def perturb_axis(a: np.ndarray, max_angle: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate `a` by a random angle up to max_angle about a random perpendicular axis."""
    perp = np.cross(a, random_unit(rng))
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(a, [1.0, 0, 0])
    perp /= np.linalg.norm(perp)
    ang = rng.uniform(0, max_angle)
    return rotvec_to_mat(perp * ang) @ a


def z_clearance(a_palm: np.ndarray, r: float, h: float) -> float:
    """Height of the object centre above the palm face when it just touches it."""
    c = abs(a_palm[2])
    s = np.sqrt(max(0.0, 1 - c * c))
    return r * s + (h / 2) * c
