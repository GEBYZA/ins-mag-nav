"""Quaternion utilities: Hamilton convention q = [w, x, y, z], body -> NED rotation."""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_multiply(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Hamilton product q ⊗ p."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_delta_angle(dtheta: np.ndarray) -> np.ndarray:
    """Small rotation delta in body frame -> quaternion (approx)."""
    half = 0.5 * dtheta
    th = np.linalg.norm(half)
    if th < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    s = np.sin(th) / th
    return quat_normalize(np.array([np.cos(th), half[0] * s, half[1] * s, half[2] * s]))


def quat_to_rot_b_to_n(q: np.ndarray) -> np.ndarray:
    """Rotation C_bn: v_ned = C_bn @ v_body."""
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def euler_ned_to_quat(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Roll/pitch/yaw (rad) in NED -> quaternion body-to-NED."""
    cr, sr = np.cos(roll_rad / 2), np.sin(roll_rad / 2)
    cp, sp = np.cos(pitch_rad / 2), np.sin(pitch_rad / 2)
    cy, sy = np.cos(yaw_rad / 2), np.sin(yaw_rad / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return quat_normalize(np.array([w, x, y, z], dtype=float))


def quat_to_euler_ned(q: np.ndarray) -> tuple[float, float, float]:
    """Quaternion body-to-NED -> (roll, pitch, yaw) in rad."""
    c = quat_to_rot_b_to_n(q)
    roll = np.arctan2(c[2, 1], c[2, 2])
    pitch = -np.arcsin(np.clip(c[2, 0], -1.0, 1.0))
    yaw = np.arctan2(c[1, 0], c[0, 0])
    return float(roll), float(pitch), float(yaw)


def integrate_quat_gyro(q: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
    """Strapdown gyro integration: q_{k+1} = q_k ⊗ q(ω dt)."""
    w = float(np.linalg.norm(gyro))
    if w < 1e-12:
        return quat_normalize(q)
    half_angle = 0.5 * w * dt
    s = np.sin(half_angle) / w
    dq = np.array([np.cos(half_angle), gyro[0] * s, gyro[1] * s, gyro[2] * s])
    return quat_normalize(quat_multiply(q, dq))
