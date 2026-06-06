"""
Read IMU+mag CSV and run fusion. Plot optional if matplotlib installed.

Usage:
  python -m ins_mag_nav.run_from_csv ins_mag_nav/data/example_static.csv
  python -m ins_mag_nav.run_from_csv ins_mag_nav/data/example_turn.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from .navigator import InsMagNavigator

DIP = np.deg2rad(60.0)
MAG_NED = np.array([np.cos(DIP), 0.0, np.sin(DIP)])
DT = 0.01
MAG_EVERY = 5


def load_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("gx"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            vals = list(map(float, parts[:9]))
            rows.append(
                {
                    "gx": vals[0],
                    "gy": vals[1],
                    "gz": vals[2],
                    "ax": vals[3],
                    "ay": vals[4],
                    "az": vals[5],
                    "mx": vals[6],
                    "my": vals[7],
                    "mz": vals[8],
                }
            )
    return rows


def run_file(csv_path: Path, mag_every: int = MAG_EVERY) -> None:
    rows = load_csv(csv_path)
    if not rows:
        print("No data rows in", csv_path)
        return

    nav = InsMagNavigator(dt_imu=DT, mag_field_ned=MAG_NED)
    nav.initialize(q0=np.array([1.0, 0.0, 0.0, 0.0]), p0=np.zeros(3), v0=np.zeros(3))

    positions = []
    yaws = []

    for k, r in enumerate(rows):
        gyro = np.array([r["gx"], r["gy"], r["gz"]])
        accel = np.array([r["ax"], r["ay"], r["az"]])
        mag = np.array([r["mx"], r["my"], r["mz"]])
        nav.step_imu(gyro, accel, update_accel_ekf=(k % 2 == 0))
        if k % mag_every == 0:
            nav.step_mag(mag)
        st = nav.get_state()
        positions.append(st.position_ned.copy())
        from .quaternion import quat_to_rot_b_to_n

        C = quat_to_rot_b_to_n(st.quaternion_body_to_ned)
        yaws.append(float(np.arctan2(C[1, 0], C[0, 0])))

    positions = np.array(positions)
    yaws = np.array(yaws)
    print("File:", csv_path.name)
    print("  samples:", len(rows), "  duration: %.1f s" % (len(rows) * DT))
    print("  final position NED (m):", np.round(positions[-1], 3))
    print("  final yaw (deg):", round(np.rad2deg(yaws[-1]), 2))
    print("  yaw change (deg):", round(np.rad2deg(yaws[-1] - yaws[0]), 2))

    try:
        import matplotlib.pyplot as plt

        t = np.arange(len(rows)) * DT
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(t, positions[:, 0], label="North")
        ax[0].plot(t, positions[:, 1], label="East")
        ax[0].set_xlabel("t (s)")
        ax[0].set_ylabel("position (m)")
        ax[0].legend()
        ax[0].set_title(csv_path.stem)
        ax[1].plot(t, np.rad2deg(yaws))
        ax[1].set_xlabel("t (s)")
        ax[1].set_ylabel("yaw (deg)")
        ax[1].grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("  (install matplotlib to see plots)")


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    if not argv:
        data = Path(__file__).resolve().parent / "data"
        for name in ("example_static.csv", "example_turn.csv", "example_forward.csv"):
            p = data / name
            if p.exists():
                run_file(p)
            else:
                print("Missing", p, "- run: python -m ins_mag_nav.generate_examples")
        return
    run_file(Path(argv[0]))


if __name__ == "__main__":
    main()
