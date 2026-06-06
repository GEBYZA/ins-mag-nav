"""Generate example CSV datasets. Run: python -m ins_mag_nav.generate_examples"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .quaternion import integrate_quat_gyro, quat_normalize, quat_to_rot_b_to_n

DATA_DIR = Path(__file__).resolve().parent / "data"
DIP = np.deg2rad(60.0)
M_N = np.array([np.cos(DIP), 0.0, np.sin(DIP)])
G_N = np.array([0.0, 0.0, 9.81])
DT = 0.01


def _write_csv(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gx", "gy", "gz", "ax", "ay", "az", "mx", "my", "mz"])
        w.writerows(rows)


def _simulate(
    steps: int,
    turn_rate: float,
    forward_accel: float,
    noise: bool,
) -> list[list[float]]:
    rng = np.random.default_rng(42)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    rows: list[list[float]] = []
    for _ in range(steps):
        gyro = np.array([0.0, 0.0, turn_rate])
        C = quat_to_rot_b_to_n(q)
        a_n = np.array([forward_accel, 0.0, 0.0])
        accel = C.T @ (a_n - G_N)
        mag = C.T @ M_N
        mag = mag / np.linalg.norm(mag) * 40.0
        if noise:
            gyro = gyro + rng.normal(0, 0.002, 3)
            accel = accel + rng.normal(0, 0.03, 3)
            mag = mag + rng.normal(0, 0.5, 3)
        rows.append([*gyro, *accel, *mag.tolist()])
        q = integrate_quat_gyro(q, gyro, DT)
        q = quat_normalize(q)
    return rows


def main() -> None:
    # 1) 静止 3s
    static = []
    mag = [20.0, 0.0, 35.0]
    for _ in range(int(3.0 / DT)):
        static.append([0.0, 0.0, 0.0, 0.0, 0.0, -9.81, *mag])
    _write_csv(DATA_DIR / "example_static.csv", static)

    # 2) 匀速转弯 10s，约 3 deg/s
    turn = _simulate(int(10.0 / DT), np.deg2rad(3.0), 0.0, noise=True)
    _write_csv(DATA_DIR / "example_turn.csv", turn)

    # 3) 直线加速 + 轻微转弯 8s
    forward = _simulate(int(8.0 / DT), np.deg2rad(0.5), 0.3, noise=True)
    _write_csv(DATA_DIR / "example_forward.csv", forward)

    print("Wrote:", DATA_DIR / "example_static.csv")
    print("Wrote:", DATA_DIR / "example_turn.csv")
    print("Wrote:", DATA_DIR / "example_forward.csv")


if __name__ == "__main__":
    main()
