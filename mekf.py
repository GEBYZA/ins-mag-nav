"""
Multiplicative EKF (9 states): attitude error δθ, gyro bias b_g, accel bias b_a.
Updates: accelerometer (gravity direction) + magnetometer (field direction) in NED.
"""

from __future__ import annotations

import numpy as np

from .quaternion import (
    integrate_quat_gyro,
    normalize,
    quat_from_delta_angle,
    quat_multiply,
    quat_normalize,
    quat_to_rot_b_to_n,
    skew,
)


class MEKF9:
    def __init__(
        self,
        dt: float,
        gravity_ned: np.ndarray | None = None,
        mag_field_ned: np.ndarray | None = None,
        sigma_gyro: float = 0.02,
        sigma_gyro_bias: float = 1e-4,
        sigma_accel: float = 0.3,
        sigma_accel_bias: float = 1e-4,
        sigma_mag: float = 0.05,
    ) -> None:
        self.dt = float(dt)
        # NED: gravity points +Down; accelerometer at rest reads specific force ~ -g (Up)
        self.g_n = np.array([0.0, 0.0, 9.81], dtype=float) if gravity_ned is None else gravity_ned.astype(float)
        self.g_unit_n = self.g_n / max(np.linalg.norm(self.g_n), 1e-9)
        # Reference magnetic direction in NED (normalized). User should set declination/dip for site.
        if mag_field_ned is None:
            dip_deg = 60.0
            d = np.deg2rad(dip_deg)
            self.m_unit_n = np.array([np.cos(d), 0.0, np.sin(d)], dtype=float)
        else:
            self.m_unit_n = mag_field_ned.astype(float)
        self.m_unit_n = self.m_unit_n / max(np.linalg.norm(self.m_unit_n), 1e-9)

        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.x = np.zeros(9, dtype=float)  # [dtheta, bg, ba]
        self.P = np.eye(9, dtype=float) * 0.01

        # Process noise (tunable)
        self.Q = np.zeros((9, 9), dtype=float)
        self.Q[0:3, 0:3] = np.eye(3) * (sigma_gyro**2) * dt
        self.Q[3:6, 3:6] = np.eye(3) * (sigma_gyro_bias**2) * dt
        self.Q[6:9, 6:9] = np.eye(3) * (sigma_accel_bias**2) * dt

        self.R_acc = np.eye(3) * (sigma_accel**2)
        self.R_mag = np.eye(3) * (sigma_mag**2)

    def reset_attitude(self, q: np.ndarray) -> None:
        self.q = quat_normalize(np.asarray(q, dtype=float).copy())
        self.x[:] = 0.0

    def _C_bn(self) -> np.ndarray:
        return quat_to_rot_b_to_n(self.q)

    def predict(self, gyro_meas: np.ndarray, accel_meas: np.ndarray | None = None) -> None:
        """Propagate quaternion with bias-corrected gyro; covariance with error-state F."""
        dt = self.dt
        bg = self.x[3:6]
        omega = gyro_meas - bg
        self.q = integrate_quat_gyro(self.q, omega, dt)

        w = omega
        F = np.eye(9, dtype=float)
        F[0:3, 0:3] = np.eye(3) - skew(w) * dt
        F[0:3, 3:6] = -np.eye(3) * dt
        self.P = F @ self.P @ F.T + self.Q

    def _inject_error(self) -> None:
        dtheta = self.x[0:3]
        dq = quat_from_delta_angle(dtheta)
        self.q = quat_normalize(quat_multiply(self.q, dq))
        self.x[0:3] = 0.0

    def update_accel(self, accel_meas: np.ndarray) -> None:
        """Use specific force direction (gravity when low dynamics)."""
        C = self._C_bn()
        # Predicted accelerometer (specific force) direction in body: a_b ≈ C^T (-g_n/||g||)
        h = C.T @ (-self.g_unit_n)
        h = normalize(h)
        z = normalize(accel_meas)

        H = np.zeros((3, 9), dtype=float)
        H[0:3, 0:3] = skew(h)
        H[0:3, 6:9] = np.eye(3)

        y = z - h
        S = H @ self.P @ H.T + self.R_acc
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(9)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_acc @ K.T
        self._inject_error()

    def update_mag(self, mag_meas: np.ndarray) -> None:
        """Field direction in body should match C^T m_n."""
        C = self._C_bn()
        h = C.T @ self.m_unit_n
        h = normalize(h)
        z = normalize(mag_meas)

        H = np.zeros((3, 9), dtype=float)
        H[0:3, 0:3] = skew(h)

        y = z - h
        S = H @ self.P @ H.T + self.R_mag
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(9)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R_mag @ K.T
        self._inject_error()

    @property
    def gyro_bias(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def accel_bias(self) -> np.ndarray:
        return self.x[6:9].copy()
