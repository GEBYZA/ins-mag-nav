"""
Pure-Python World Magnetic Model (WMM) field calculator.
Uses NOAA WMM.COF Gauss coefficients (default: data/WMM2020.COF).

Output: geomagnetic field in NED (nT): X=North, Y=East, Z=Down.
Source: US/UK World Magnetic Model 2020-2025 (NOAA NCEI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MAX_DEG = 12
A = 6378137.0
B = 6356752.3142
MU0_4PI = 1e-7 * 4 * np.pi * 1e9  # nT scaling


def default_cof_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "WMM2020.COF"


def parse_cof(path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.zeros((MAX_DEG + 1, MAX_DEG + 1))
    h = np.zeros((MAX_DEG + 1, MAX_DEG + 1))
    dg = np.zeros((MAX_DEG + 1, MAX_DEG + 1))
    dh = np.zeros((MAX_DEG + 1, MAX_DEG + 1))
    epoch = 2020.0
    with path.open(encoding="utf-8") as f:
        first = f.readline().split()
        epoch = float(first[0])
        for line in f:
            line = line.strip()
            if not line or line[0] in "fF":
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            n, m = int(parts[0]), int(parts[1])
            if n > MAX_DEG:
                continue
            g[n, m] = float(parts[2])
            h[n, m] = float(parts[3])
            dg[n, m] = float(parts[4])
            dh[n, m] = float(parts[5])
    return epoch, g, h, dg, dh


def _legendre(theta: float, nmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Schmidt quasi-normalized Legendre P_n^m(cos theta), sin(theta) form."""
    x = np.cos(theta)
    p = np.zeros((nmax + 1, nmax + 1))
    dp = np.zeros((nmax + 1, nmax + 1))
    p[0, 0] = 1.0
    dp[0, 0] = 0.0
    if nmax == 0:
        return p, dp
    p[1, 1] = np.sqrt(1 - x * x)
    dp[1, 1] = x
    for n in range(1, nmax + 1):
        p[n, n] = (2 * n - 1) * p[n - 1, n - 1] * np.sqrt(max(1 - x * x, 0))
        dp[n, n] = n * x * p[n, n]
        for m in range(0, n):
            if n == 1 and m == 0:
                p[n, m] = x * p[0, 0]
                dp[n, m] = p[0, 0]
            else:
                f = np.sqrt((2 * n - 1) / (n - m)) if n != m else 1.0
                g = np.sqrt((n + m - 1) / (n - m)) if (n - m) > 1 else 1.0
                p[n, m] = f * x * p[n - 1, m] - g * p[n - 2, m]
                dp[n, m] = f * (x * dp[n - 1, m] - p[n - 1, m]) - g * dp[n - 2, m]
    return p, dp


def wmm_field_ned(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    decimal_year: float,
    epoch: float,
    g: np.ndarray,
    h: np.ndarray,
    dg: np.ndarray,
    dh: np.ndarray,
) -> np.ndarray:
    """Compute WMM field (North, East, Down) in nT."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    dt = decimal_year - epoch
    gc = g + dg * dt
    hc = h + dh * dt

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    # geocentric radius and colatitude
    r = np.sqrt((A * A * cos_lat * cos_lat + B * B * sin_lat * sin_lat))
    r += alt_m
    theta = np.pi / 2 - lat  # approx geocentric colatitude for mid-latitudes
    if abs(lat) < np.deg2rad(89):
        # better geodetic to geocentric colatitude
        geoc_lat = np.arctan((B * B / (A * A)) * np.tan(lat))
        theta = np.pi / 2 - geoc_lat
    r_n = (A * A * cos_lat) / np.sqrt(A * A * cos_lat * cos_lat + B * B * sin_lat * sin_lat) + alt_m

    p, dp = _legendre(theta, MAX_DEG)
    cos_mlon = np.cos(np.arange(MAX_DEG + 1) * lon)
    sin_mlon = np.sin(np.arange(MAX_DEG + 1) * lon)

    br = bt = bp = 0.0
    for n in range(1, MAX_DEG + 1):
        rr = (6371200.0 / r_n) ** (n + 2)
        for m in range(0, n + 1):
            pm = p[n, m]
            dpm = dp[n, m]
            gnm = gc[n, m]
            hnm = hc[n, m] if m > 0 else 0.0
            if m == 0:
                br += (n + 1) * rr * gnm * pm
                bt -= rr * gnm * dpm
            else:
                cosml = cos_mlon[m]
                sinml = sin_mlon[m]
                br += (n + 1) * rr * (gnm * cosml + hnm * sinml) * pm
                bt -= rr * (gnm * cosml + hnm * sinml) * dpm
                bp += rr * m * (gnm * sinml - hnm * cosml) * pm / max(np.sin(theta), 1e-6)

    # geocentric spherical to geodetic NED
    psi = theta - (np.pi / 2 - lat)
    xn = -bt * np.cos(psi) - br * np.sin(psi)
    ye = bp
    zd = bt * np.sin(psi) - br * np.cos(psi)
    return np.array([xn, ye, zd], dtype=float)


class WMMModel:
    """Global Earth magnetic model (NOAA WMM)."""

    def __init__(self, cof_path: Path | str | None = None) -> None:
        path = Path(cof_path) if cof_path else default_cof_path()
        if not path.exists():
            raise FileNotFoundError(f"WMM coefficient file not found: {path}")
        self.epoch, self.g, self.h, self.dg, self.dh = parse_cof(path)
        self.cof_path = path

    def field_ned(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0, decimal_year: float | None = None) -> np.ndarray:
        if decimal_year is None:
            decimal_year = self.epoch
        return wmm_field_ned(lat_deg, lon_deg, alt_m, decimal_year, self.epoch, self.g, self.h, self.dg, self.dh)

    def field_gradient_ned_per_m(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
        dlat = 0.01
        dlon = 0.01
        latr = np.deg2rad(lat_deg)
        rm = 6378137.0 + alt_m
        dn = np.deg2rad(dlat) * rm
        de = np.deg2rad(dlon) * rm * max(np.cos(latr), 1e-6)
        G = np.zeros((3, 3))
        G[:, 0] = (self.field_ned(lat_deg + dlat, lon_deg, alt_m) - self.field_ned(lat_deg - dlat, lon_deg, alt_m)) / (2 * dn)
        G[:, 1] = (self.field_ned(lat_deg, lon_deg + dlon, alt_m) - self.field_ned(lat_deg, lon_deg - dlon, alt_m)) / (2 * de)
        return G
