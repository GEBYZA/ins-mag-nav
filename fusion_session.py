"""
松组合 INS + 地磁图融合 API。

初始: 经纬高、姿态、速度
输入: 多帧九轴 (gyro rad/s, accel m/s², mag uT)
输出: 末位置、末姿态、末速度、末加速度 (NED)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geodetic import ned_to_geodetic
from .global_mag_model import GlobalMagneticModel, load_global_model
from .loose_coupled_ekf import LooseCoupledEKF
from .quaternion import euler_ned_to_quat, quat_to_euler_ned


@dataclass
class InitialState:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    velocity_north_mps: float = 0.0
    velocity_east_mps: float = 0.0
    velocity_down_mps: float = 0.0


@dataclass
class ImuMagSample:
    gx: float
    gy: float
    gz: float
    ax: float
    ay: float
    az: float
    mx: float
    my: float
    mz: float

    @classmethod
    def from_sequence(cls, values: list[float] | tuple[float, ...]) -> ImuMagSample:
        if len(values) != 9:
            raise ValueError("每帧需要 9 个数: gx,gy,gz, ax,ay,az, mx,my,mz")
        return cls(*[float(v) for v in values])

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array([self.gx, self.gy, self.gz]),
            np.array([self.ax, self.ay, self.az]),
            np.array([self.mx, self.my, self.mz]),
        )


@dataclass
class NavigationResult:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    velocity_north_mps: float
    velocity_east_mps: float
    velocity_down_mps: float
    accel_north_mps2: float
    accel_east_mps2: float
    accel_down_mps2: float
    position_north_m: float
    position_east_m: float
    position_down_m: float
    mag_updates_accepted: int = 0
    mag_updates_rejected: int = 0
    num_samples: int = 0
    quaternion: np.ndarray = field(repr=False)
    gyro_bias: np.ndarray = field(repr=False)
    accel_bias: np.ndarray = field(repr=False)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("quaternion", "gyro_bias", "accel_bias")}


class FusionSession:
    """松组合地磁图 + INS 融合会话。"""

    def __init__(
        self,
        dt_imu: float,
        mag_map: GlobalMagneticModel | None = None,
        use_global_wmm: bool = True,
        **ekf_kwargs,
    ) -> None:
        self.dt = float(dt_imu)
        if mag_map is not None:
            self._map = mag_map
        elif use_global_wmm:
            self._map = load_global_model()
        else:
            from .magnetic_map import load_or_create_default_map

            self._map = load_or_create_default_map()
        self._ekf_kw = ekf_kwargs
        self._initial: InitialState | None = None
        self._ekf: LooseCoupledEKF | None = None
        self._sample_count = 0

    def set_initial(self, init: InitialState) -> None:
        self._initial = init
        self._ekf = LooseCoupledEKF(
            self.dt,
            self._map,
            init.latitude_deg,
            init.longitude_deg,
            init.altitude_m,
            **self._ekf_kw,
        )
        q0 = euler_ned_to_quat(
            np.deg2rad(init.roll_deg),
            np.deg2rad(init.pitch_deg),
            np.deg2rad(init.yaw_deg),
        )
        v0 = np.array([init.velocity_north_mps, init.velocity_east_mps, init.velocity_down_mps])
        self._ekf.set_initial(q0, v0, np.zeros(3))
        self._sample_count = 0

    def run_sample(self, sample: ImuMagSample, use_acc: bool = True) -> None:
        if self._ekf is None:
            raise RuntimeError("请先 set_initial()")
        g, a, m = sample.to_arrays()
        self._ekf.step(g, a, m, use_acc=use_acc)
        self._sample_count += 1

    def run_samples(
        self,
        samples: list[ImuMagSample] | list[tuple[float, ...] | list[float]],
        use_acc: bool = True,
    ) -> NavigationResult:
        for s in samples:
            if not isinstance(s, ImuMagSample):
                s = ImuMagSample.from_sequence(s)
            self.run_sample(s, use_acc=use_acc)
        return self.get_result()

    def get_result(self) -> NavigationResult:
        if self._ekf is None or self._initial is None:
            raise RuntimeError("未初始化")
        init = self._initial
        ekf = self._ekf
        lat, lon, alt = ned_to_geodetic(ekf.p, init.latitude_deg, init.longitude_deg, init.altitude_m)
        roll, pitch, yaw = quat_to_euler_ned(ekf.q)
        return NavigationResult(
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_m=alt,
            roll_deg=float(np.rad2deg(roll)),
            pitch_deg=float(np.rad2deg(pitch)),
            yaw_deg=float(np.rad2deg(yaw)),
            velocity_north_mps=float(ekf.v[0]),
            velocity_east_mps=float(ekf.v[1]),
            velocity_down_mps=float(ekf.v[2]),
            accel_north_mps2=float(ekf.a_n[0]),
            accel_east_mps2=float(ekf.a_n[1]),
            accel_down_mps2=float(ekf.a_n[2]),
            position_north_m=float(ekf.p[0]),
            position_east_m=float(ekf.p[1]),
            position_down_m=float(ekf.p[2]),
            mag_updates_accepted=ekf.stats["mag_accept"],
            mag_updates_rejected=ekf.stats["mag_reject"],
            num_samples=self._sample_count,
            quaternion=ekf.q.copy(),
            gyro_bias=ekf.bg.copy(),
            accel_bias=ekf.ba.copy(),
        )
