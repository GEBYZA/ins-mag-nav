"""
松组合 INS+地磁图融合示例。

  python -m ins_mag_nav.run_user_example
  python -m ins_mag_nav.run_user_example ins_mag_nav/data/mission_example.txt
  python -m ins_mag_nav.magnetic_map   # 生成地磁图
"""

from __future__ import annotations

import sys
from pathlib import Path

from .fusion_session import FusionSession, ImuMagSample, InitialState


def parse_mission_file(path: Path) -> tuple[InitialState, list[ImuMagSample], float]:
    init: InitialState | None = None
    samples: list[ImuMagSample] = []
    dt = 0.01
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("dt="):
            dt = float(line.split("=", 1)[1])
            continue
        if line.lower().startswith("init:"):
            v = [float(x) for x in line.split(":", 1)[1].split(",")]
            init = InitialState(
                latitude_deg=v[0],
                longitude_deg=v[1],
                altitude_m=v[2],
                roll_deg=v[3],
                pitch_deg=v[4],
                yaw_deg=v[5],
                velocity_north_mps=v[6],
                velocity_east_mps=v[7],
                velocity_down_mps=v[8],
            )
            continue
        parts = [float(x) for x in line.replace(",", " ").split()]
        if len(parts) >= 9:
            samples.append(ImuMagSample.from_sequence(parts[:9]))
    if init is None:
        raise ValueError("缺少 init: 行")
    return init, samples, dt


def print_result(init: InitialState, r, dt: float, n: int) -> None:
    print("=" * 56)
    print("【初始】")
    print(f"  位置 {init.latitude_deg:.6f}, {init.longitude_deg:.6f}, {init.altitude_m:.1f} m")
    print(f"  姿态 roll={init.roll_deg} pitch={init.pitch_deg} yaw={init.yaw_deg} deg")
    print(f"  速度 vn={init.velocity_north_mps} ve={init.velocity_east_mps} vd={init.velocity_down_mps} m/s")
    print(f"  帧数 {n}  dt={dt}s  时长={n*dt:.2f}s")
    print("-" * 56)
    print("【末状态 — 松组合 EKF + 地磁图】")
    print(f"  位置 {r.latitude_deg:.8f}, {r.longitude_deg:.8f}, {r.altitude_m:.3f} m")
    print(f"  NED位移 北{r.position_north_m:.4f} 东{r.position_east_m:.4f} 地{r.position_down_m:.4f} m")
    print(f"  姿态 roll={r.roll_deg:.3f} pitch={r.pitch_deg:.3f} yaw={r.yaw_deg:.3f} deg")
    print(f"  速度 vn={r.velocity_north_mps:.4f} ve={r.velocity_east_mps:.4f} vd={r.velocity_down_mps:.4f} m/s")
    print(f"  加速度 an={r.accel_north_mps2:.4f} ae={r.accel_east_mps2:.4f} ad={r.accel_down_mps2:.4f} m/s²")
    print(f"  地磁更新 接受={r.mag_updates_accepted} 拒绝={r.mag_updates_rejected}")
    print("=" * 56)


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        init, samples, dt = parse_mission_file(path)
    else:
        path = Path(__file__).parent / "data" / "mission_example.txt"
        if path.exists():
            init, samples, dt = parse_mission_file(path)
        else:
            init = InitialState(39.9042, 116.4074, 50.0)
            samples = [ImuMagSample(0, 0, 0.052, 0, 0, -9.81, 20, 0, 35) for _ in range(50)]
            dt = 0.01

    session = FusionSession(dt_imu=dt)
    session.set_initial(init)
    result = session.run_samples(samples)
    print_result(init, result, dt, len(samples))


if __name__ == "__main__":
    main()
