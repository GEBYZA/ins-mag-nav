"""
High-level fusion: MEKF attitude (gyro + accel + mag) + strapdown position/velocity.
Call `step_imu` every IMU sample; optionally `step_mag` at mag rate (often lower).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mekf import MEKF9
from .strapdown import StrapdownINS


@dataclass
class NavState:
    position_ned: np.ndarray
    velocity_ned: np.ndarray
    quaternion_body_to_ned: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray


class InsMagNavigator:
    def __init__(
        self,
        dt_imu: float,
        gravity_ned: np.ndarray | None = None,
        mag_field_ned: np.ndarray | None = None,
        **mekf_kwargs,
    ) -> None:
        self.dt = float(dt_imu)
        self.ekf = MEKF9(self.dt, gravity_ned=gravity_ned, mag_field_ned=mag_field_ned, **mekf_kwargs)
        self.ins = StrapdownINS(gravity_ned=gravity_ned)
        self._use_mag = True

    def initialize(self, q0: np.ndarray, p0: np.ndarray | None = None, v0: np.ndarray | None = None) -> None:
        self.ekf.reset_attitude(q0)
        if p0 is not None:
            self.ins.p = np.asarray(p0, dtype=float).copy()
        if v0 is not None:
            self.ins.v = np.asarray(v0, dtype=float).copy()

    def step_imu(self, gyro: np.ndarray, accel: np.ndarray, update_accel_ekf: bool = True) -> NavState:
        self.ekf.predict(gyro, accel)
        if update_accel_ekf:
            self.ekf.update_accel(accel)
        self.ins.step(self.ekf.q, accel, self.ekf.accel_bias, self.dt)
        return self.get_state()

    def step_mag(self, mag: np.ndarray) -> NavState:
        if self._use_mag:
            self.ekf.update_mag(mag)
        return self.get_state()

    def get_state(self) -> NavState:
        return NavState(
            position_ned=self.ins.p.copy(),
            velocity_ned=self.ins.v.copy(),
            quaternion_body_to_ned=self.ekf.q.copy(),
            gyro_bias=self.ekf.gyro_bias,
            accel_bias=self.ekf.accel_bias,
        )
