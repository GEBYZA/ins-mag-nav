"""
Loosely-coupled INS / geomagnetic-map EKF (15 error states).

Error state: [dPn, dPe, dPd, dVn, dVe, dVd, dPhi_n, dPhi_e, dPhi_d, bg(3), ba(3)]
Nominal: p_ned, v_ned, q_bn, gyro_bias, accel_bias

松组合:
  - INS strapdown propagates nominal state
  - Accel update: gravity direction (gated, low dynamics)
  - Mag-map update: m_b vs C^T * B_map(lat,lon) with position + attitude coupling
"""

from __future__ import annotations

import numpy as np

from collections import deque

from .geodetic import ned_to_geodetic
from .quaternion import (
    integrate_quat_gyro,
    normalize,
    quat_from_delta_angle,
    quat_multiply,
    quat_normalize,
    quat_to_euler_ned,
    quat_to_rot_b_to_n,
    skew,
)

def _supports_alt(mag_map) -> bool:
    import inspect

    sig = inspect.signature(mag_map.field_at)
    return "alt" in sig.parameters


N_ERR = 15
IDX_P = slice(0, 3)
IDX_V = slice(3, 6)
IDX_A = slice(6, 9)
IDX_BG = slice(9, 12)
IDX_BA = slice(12, 15)


class LooseCoupledEKF:
    def __init__(
        self,
        dt: float,
        mag_map,
        lat0_deg: float,
        lon0_deg: float,
        alt0_m: float,
        sigma_gyro: float = 0.015,
        sigma_accel: float = 0.25,
        sigma_mag: float = 0.8,
        sigma_gyro_bias: float = 1e-4,
        sigma_accel_bias: float = 1e-4,
        nis_gate: float = 11.34,
        mag_median_window: int = 5,
        mag_strength_gate: float = 0.35,
    ) -> None:
        self.dt = float(dt)
        self.map = mag_map
        self.lat0 = float(lat0_deg)
        self.lon0 = float(lon0_deg)
        self.alt0 = float(alt0_m)
        self.g_n = np.array([0.0, 0.0, 9.81])
        self.nis_gate = float(nis_gate)

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.a_n = np.zeros(3)

        self.dx = np.zeros(N_ERR)
        self.P = np.diag(
            [1.0, 1.0, 2.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]
        )

        qsg = sigma_gyro**2 * dt
        qsa = sigma_accel**2 * dt
        self.Q = np.zeros((N_ERR, N_ERR))
        self.Q[IDX_A, IDX_A] = np.eye(3) * qsg
        self.Q[IDX_BG, IDX_BG] = np.eye(3) * (sigma_gyro_bias**2 * dt)
        self.Q[IDX_BA, IDX_BA] = np.eye(3) * (sigma_accel_bias**2 * dt)
        self.Q[IDX_V, IDX_V] = np.eye(3) * qsa * 0.1
        self.R_acc = np.eye(3) * (sigma_accel**2)
        self.R_mag = np.eye(3) * (sigma_mag**2)
        self.stats = {"mag_accept": 0, "mag_reject": 0, "acc_skip": 0}
        self._mag_buf: deque[np.ndarray] = deque(maxlen=max(1, mag_median_window))
        self._mag_strength_gate = mag_strength_gate

    def set_initial(
        self,
        q0: np.ndarray,
        v0: np.ndarray | None = None,
        p0: np.ndarray | None = None,
    ) -> None:
        self.q = quat_normalize(np.asarray(q0, dtype=float))
        self.v = np.zeros(3) if v0 is None else np.asarray(v0, dtype=float).copy()
        self.p = np.zeros(3) if p0 is None else np.asarray(p0, dtype=float).copy()
        self.bg[:] = 0.0
        self.ba[:] = 0.0
        self.dx[:] = 0.0

    def _lla(self) -> tuple[float, float, float]:
        return ned_to_geodetic(self.p, self.lat0, self.lon0, self.alt0)

    def _inject(self) -> None:
        self.p += self.dx[IDX_P]
        self.v += self.dx[IDX_V]
        dq = quat_from_delta_angle(self.dx[IDX_A])
        self.q = quat_normalize(quat_multiply(self.q, dq))
        self.bg += self.dx[IDX_BG]
        self.ba += self.dx[IDX_BA]
        self.dx[:] = 0.0

    def _ekf_update(self, y: np.ndarray, H: np.ndarray, R: np.ndarray) -> bool:
        # Huber-like down-weight large innovations
        y_norm = float(np.linalg.norm(y))
        huber = 1.0 if y_norm < 1.5 else max(0.2, 1.5 / y_norm)
        y = y * huber
        S = H @ self.P @ H.T + R
        try:
            nis = float(y.T @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            return False
        if nis > self.nis_gate:
            return False
        scale = min(8.0, max(1.0, nis / max(len(y), 1)))
        R_eff = R * scale
        S = H @ self.P @ H.T + R_eff
        K = self.P @ H.T @ np.linalg.inv(S)
        self.dx = self.dx + K @ y
        I = np.eye(N_ERR)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_eff @ K.T
        self._inject()
        return True

    def predict(self, gyro: np.ndarray, accel: np.ndarray) -> None:
        dt = self.dt
        w = gyro - self.bg
        self.q = integrate_quat_gyro(self.q, w, dt)
        C = quat_to_rot_b_to_n(self.q)
        f_b = accel - self.ba
        self.a_n = C @ f_b + self.g_n
        self.v = self.v + self.a_n * dt
        self.p = self.p + self.v * dt

        F = np.eye(N_ERR)
        F[IDX_P, IDX_V] = np.eye(3) * dt
        F[IDX_A, IDX_A] = np.eye(3) - skew(w) * dt
        F[IDX_A, IDX_BG] = -np.eye(3) * dt
        F[IDX_V, IDX_A] = -skew(C @ f_b) * dt
        F[IDX_V, IDX_BA] = -C * dt
        self.P = F @ self.P @ F.T + self.Q

    def update_accel(self, accel: np.ndarray) -> None:
        if abs(np.linalg.norm(accel) - 9.81) > 2.5:
            self.stats["acc_skip"] += 1
            return
        C = quat_to_rot_b_to_n(self.q)
        h = normalize(C.T @ (-self.g_n / 9.81))
        z = normalize(accel)
        y = z - h
        H = np.zeros((3, N_ERR))
        H[:, IDX_A] = skew(h)
        H[:, IDX_BA] = np.eye(3)
        self._ekf_update(y, H, self.R_acc)

    def update_mag_map(self, mag: np.ndarray) -> None:
        self._mag_buf.append(np.asarray(mag, dtype=float))
        if len(self._mag_buf) >= 3:
            stack = np.stack(list(self._mag_buf), axis=0)
            mag_use = np.median(stack, axis=0)
        else:
            mag_use = mag

        m_norm = np.linalg.norm(mag_use)
        if m_norm < 1e-6:
            self.stats["mag_reject"] += 1
            return
        lat, lon, alt = self._lla()
        B_n = self.map.field_at(lat, lon, alt) if _supports_alt(self.map) else self.map.field_at(lat, lon)
        B_norm = np.linalg.norm(B_n)
        if B_norm < 1e-6:
            self.stats["mag_reject"] += 1
            return
        if abs(m_norm - B_norm) / max(B_norm, 1.0) > self._mag_strength_gate:
            self.stats["mag_reject"] += 1
            return
        C = quat_to_rot_b_to_n(self.q)
        h = C.T @ B_n
        y = mag_use - h

        latr = np.deg2rad(lat)
        rm = 6378137.0 + alt
        dlat_dn = np.rad2deg(1.0 / rm)
        dlon_de = np.rad2deg(1.0 / (rm * max(np.cos(latr), 1e-6)))
        if _supports_alt(self.map):
            dB_dp = self.map.field_gradient_ned_per_m(lat, lon, alt)
        else:
            dB_dp = self.map.field_gradient_ned_per_m(lat, lon, self.alt0)
        dB_dlat = dB_dp[:, 0] * dlat_dn
        dB_dlon = dB_dp[:, 1] * dlon_de
        J_bn = np.column_stack([dB_dlat, dB_dlon, np.zeros(3)])
        H = np.zeros((3, N_ERR))
        H[:, IDX_A] = -skew(h)
        H[:, IDX_P] = C.T @ J_bn

        ok = self._ekf_update(y, H, self.R_mag)
        if ok:
            self.stats["mag_accept"] += 1
        else:
            self.stats["mag_reject"] += 1

    def step(self, gyro: np.ndarray, accel: np.ndarray, mag: np.ndarray, use_acc: bool = True) -> None:
        self.predict(gyro, accel)
        if use_acc:
            self.update_accel(accel)
        self.update_mag_map(mag)

    @property
    def euler_deg(self) -> tuple[float, float, float]:
        r, p, y = quat_to_euler_ned(self.q)
        return float(np.rad2deg(r)), float(np.rad2deg(p)), float(np.rad2deg(y))
