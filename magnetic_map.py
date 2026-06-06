"""
Regional geomagnetic map: NED field vector Bn, Be, Bd (uT) on lat/lon grid.
Used by loosely-coupled mag-map matching (松组合地磁图匹配).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .geodetic import mag_ned_unit_from_inclination


class GeomagneticMap:
    """Bilinear-interpolated geomagnetic vector map."""

    def __init__(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        field_ned: np.ndarray,
        lat0_deg: float,
        lon0_deg: float,
    ) -> None:
        """
        field_ned: shape (n_lat, n_lon, 3) in uT, NED components.
        """
        self.lats = np.asarray(lats, dtype=float)
        self.lons = np.asarray(lons, dtype=float)
        self.field = np.asarray(field_ned, dtype=float)
        self.lat0 = float(lat0_deg)
        self.lon0 = float(lon0_deg)
        if self.field.ndim != 3 or self.field.shape[2] != 3:
            raise ValueError("field_ned must be (n_lat, n_lon, 3)")

    @classmethod
    def from_csv(cls, path: Path | str, lat0_deg: float | None = None, lon0_deg: float | None = None) -> GeomagneticMap:
        path = Path(path)
        rows: list[tuple[float, float, float, float, float]] = []
        with path.open(newline="", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("lat"):
                    continue
                p = [float(x) for x in line.replace(",", " ").split()]
                if len(p) >= 5:
                    rows.append((p[0], p[1], p[2], p[3], p[4]))
        if not rows:
            raise ValueError(f"empty map: {path}")
        lats = sorted({r[0] for r in rows})
        lons = sorted({r[1] for r in rows})
        ni, nj = len(lats), len(lons)
        field = np.zeros((ni, nj, 3), dtype=float)
        lut = {(r[0], r[1]): np.array([r[2], r[3], r[4]]) for r in rows}
        for i, la in enumerate(lats):
            for j, lo in enumerate(lons):
                field[i, j] = lut.get((la, lo), np.nan)
        if np.any(np.isnan(field)):
            raise ValueError("map grid incomplete")
        lat0 = lats[len(lats) // 2] if lat0_deg is None else lat0_deg
        lon0 = lons[len(lons) // 2] if lon0_deg is None else lon0_deg
        return cls(np.array(lats), np.array(lons), field, lat0, lon0)

    def _idx(self, lat: float, lon: float) -> tuple[float, float, int, int, int, int]:
        la = float(np.clip(lat, self.lats[0], self.lats[-1]))
        lo = float(np.clip(lon, self.lons[0], self.lons[-1]))
        i1 = int(np.searchsorted(self.lats, la) - 1)
        i1 = max(0, min(i1, len(self.lats) - 2))
        j1 = int(np.searchsorted(self.lons, lo) - 1)
        j1 = max(0, min(j1, len(self.lons) - 2))
        la1, la2 = self.lats[i1], self.lats[i1 + 1]
        lo1, lo2 = self.lons[j1], self.lons[j1 + 1]
        t = 0.0 if la2 == la1 else (la - la1) / (la2 - la1)
        u = 0.0 if lo2 == lo1 else (lo - lo1) / (lo2 - lo1)
        return t, u, i1, i1 + 1, j1, j1 + 1

    def field_at(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
        t, u, i0, i1, j0, j1 = self._idx(lat_deg, lon_deg)
        f00 = self.field[i0, j0]
        f01 = self.field[i0, j1]
        f10 = self.field[i1, j0]
        f11 = self.field[i1, j1]
        f0 = f00 * (1 - u) + f01 * u
        f1 = f10 * (1 - u) + f11 * u
        return f0 * (1 - t) + f1 * t

    def field_gradient_ned_per_m(
        self, lat_deg: float, lon_deg: float, alt_m: float = 0.0
    ) -> np.ndarray:
        """dB/dp_ned  (3x3), p=[N,E,D] in meters."""
        dlat = max(self.lats[1] - self.lats[0], 1e-6)
        dlon = max(self.lons[1] - self.lons[0], 1e-6)
        latr = np.deg2rad(lat_deg)
        rm = 6378137.0 + alt_m
        d_n = np.deg2rad(dlat) * rm
        d_e = np.deg2rad(dlon) * rm * max(np.cos(latr), 1e-6)
        b_p = self.field_at(lat_deg + dlat, lon_deg)
        b_m = self.field_at(lat_deg - dlat, lon_deg)
        b_pe = self.field_at(lat_deg, lon_deg + dlon)
        b_me = self.field_at(lat_deg, lon_deg - dlon)
        grad = np.zeros((3, 3), dtype=float)
        grad[:, 0] = (b_p - b_m) / (2 * d_n)
        grad[:, 1] = (b_pe - b_me) / (2 * d_e)
        grad[:, 2] = 0.0
        return grad

    def save_csv(self, path: Path | str) -> None:
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["lat_deg", "lon_deg", "Bn_uT", "Be_uT", "Bd_uT"])
            for i, la in enumerate(self.lats):
                for j, lo in enumerate(self.lons):
                    b = self.field[i, j]
                    w.writerow([f"{la:.6f}", f"{lo:.6f}", f"{b[0]:.4f}", f"{b[1]:.4f}", f"{b[2]:.4f}"])


def build_synthetic_map(
    lat_center: float = 39.9042,
    lon_center: float = 116.4074,
    span_deg: float = 0.02,
    step_deg: float = 0.002,
    inclination_deg: float = 60.0,
    declination_deg: float = -6.0,
    anomaly_amp_uT: float = 8.0,
) -> GeomagneticMap:
    """Synthetic anomaly map for demo / test."""
    lats = np.arange(lat_center - span_deg, lat_center + span_deg + 1e-9, step_deg)
    lons = np.arange(lon_center - span_deg, lon_center + span_deg + 1e-9, step_deg)
    base = mag_ned_unit_from_inclination(inclination_deg, declination_deg) * 50.0
    field = np.zeros((len(lats), len(lons), 3), dtype=float)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            dn = (la - lat_center) * 111320.0
            de = (lo - lon_center) * 111320.0 * np.cos(np.deg2rad(lat_center))
            anomaly = anomaly_amp_uT * (
                np.sin(dn / 400.0) * np.cos(de / 350.0)
                + 0.5 * np.sin(de / 200.0)
            )
            grad_n = anomaly * 0.02
            grad_e = anomaly * 0.015
            field[i, j] = base + np.array([grad_n, grad_e, anomaly * 0.3])
    return GeomagneticMap(lats, lons, field, lat_center, lon_center)


def default_map_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "mag_map_beijing.csv"


def load_or_create_default_map() -> GeomagneticMap:
    p = default_map_path()
    if p.exists():
        return GeomagneticMap.from_csv(p)
    m = build_synthetic_map()
    p.parent.mkdir(parents=True, exist_ok=True)
    m.save_csv(p)
    return m


if __name__ == "__main__":
    mp = build_synthetic_map()
    out = default_map_path()
    mp.save_csv(out)
    print("Wrote", out, "grid", mp.field.shape)
