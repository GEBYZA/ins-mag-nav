"""Strapdown INS: integrate velocity and position in NED using attitude and specific force."""

from __future__ import annotations

import numpy as np

from .quaternion import quat_to_rot_b_to_n


class StrapdownINS:
    def __init__(self, gravity_ned: np.ndarray | None = None) -> None:
        self.g_n = np.array([0.0, 0.0, 9.81], dtype=float) if gravity_ned is None else gravity_ned.astype(float)
        self.p = np.zeros(3, dtype=float)
        self.v = np.zeros(3, dtype=float)

    def step(self, q: np.ndarray, accel_body: np.ndarray, accel_bias: np.ndarray, dt: float) -> None:
        C_bn = quat_to_rot_b_to_n(q)
        f_b = accel_body - accel_bias
        a_n = C_bn @ f_b + self.g_n
        self.v = self.v + a_n * dt
        self.p = self.p + self.v * dt
