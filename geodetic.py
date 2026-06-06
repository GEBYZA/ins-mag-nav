"""WGS84 geodetic (lat/lon/alt) and local NED conversions."""

from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_geodetic(ecef: np.ndarray) -> tuple[float, float, float]:
    x, y, z = ecef
    lon = np.arctan2(y, x)
    p = np.sqrt(x * x + y * y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    alt = p / np.cos(lat) - n
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)


def rot_ecef_to_ned(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array(
        [
            [-sl * co, -sl * so, cl],
            [-so, co, 0.0],
            [-cl * co, -cl * so, -sl],
        ],
        dtype=float,
    )


def geodetic_to_ned(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> np.ndarray:
    ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    ref = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    return rot_ecef_to_ned(lat0_deg, lon0_deg) @ (ecef - ref)


def ned_to_geodetic(
    ned: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> tuple[float, float, float]:
    ref = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    r_ned_to_ecef = rot_ecef_to_ned(lat0_deg, lon0_deg).T
    return ecef_to_geodetic(ref + r_ned_to_ecef @ np.asarray(ned, dtype=float))


def mag_ned_unit_from_inclination(inclination_deg: float, declination_deg: float = 0.0) -> np.ndarray:
    """Mag field unit vector in NED from inclination + declination (deg)."""
    inc = np.deg2rad(inclination_deg)
    dec = np.deg2rad(declination_deg)
    m = np.array([np.cos(inc) * np.cos(dec), np.cos(inc) * np.sin(dec), np.sin(inc)], dtype=float)
    return m / np.linalg.norm(m)
