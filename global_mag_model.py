"""
Global geomagnetic field model for loosely-coupled navigation.

Primary: NOAA WMM2020 (coefficients in data/WMM2020.COF, worldwide valid).
Optional: precomputed 5° grid cache for fast lookup.
Download: python -m ins_mag_nav.download_wmm
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

from .wmm_pure import WMMModel, default_cof_path

WMM_DOWNLOAD_URLS = [
    "https://raw.githubusercontent.com/DMT-Services/GeoMagnetism/master/WMM.cof",
    "https://www.ncei.noaa.gov/geomag/geomag/wmm/softs/wmm2020/WMM.COF",
]


class GlobalMagneticModel:
    """Earth-wide WMM field provider (NED, nT)."""

    def __init__(self, wmm: WMMModel | None = None, grid_cache: Path | None = None) -> None:
        self.wmm = wmm or WMMModel()
        self._grid = None
        if grid_cache and grid_cache.exists():
            self._load_grid(grid_cache)

    def field_at(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
        if self._grid is not None:
            return self._interp_grid(lat_deg, lon_deg)
        return self.wmm.field_ned(lat_deg, lon_deg, alt_m)

    def field_gradient_ned_per_m(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
        return self.wmm.field_gradient_ned_per_m(lat_deg, lon_deg, alt_m)

    def _load_grid(self, path: Path) -> None:
        if path.suffix == ".npz":
            z = np.load(path)
            self._grid = (z["lats"], z["lons"], z["field"])
        elif path.suffix == ".json":
            d = json.loads(path.read_text(encoding="utf-8"))
            self._grid = (np.array(d["lats"]), np.array(d["lons"]), np.array(d["field"]))

    def _interp_grid(self, lat_deg: float, lon_deg: float) -> np.ndarray:
        lats, lons, field = self._grid
        la = float(np.clip(lat_deg, lats[0], lats[-1]))
        lo = float(np.clip(lon_deg, lons[0], lons[-1]))
        i1 = max(0, min(int(np.searchsorted(lats, la) - 1), len(lats) - 2))
        j1 = max(0, min(int(np.searchsorted(lons, lo) - 1), len(lons) - 2))
        t = (la - lats[i1]) / max(lats[i1 + 1] - lats[i1], 1e-9)
        u = (lo - lons[j1]) / max(lons[j1 + 1] - lons[j1], 1e-9)
        f00, f01, f10, f11 = field[i1, j1], field[i1, j1 + 1], field[i1 + 1, j1], field[i1 + 1, j1 + 1]
        f0 = f00 * (1 - u) + f01 * u
        f1 = f10 * (1 - u) + f11 * u
        return f0 * (1 - t) + f1 * t


def download_wmm_cof(dest: Path | None = None) -> Path:
    dest = dest or default_cof_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in WMM_DOWNLOAD_URLS:
        try:
            urllib.request.urlretrieve(url, dest)
            parse_test = WMMModel(dest)
            _ = parse_test.field_ned(40.0, -105.0, 0.0)
            return dest
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to download WMM.COF: {last_err}")


def load_global_model(use_grid_cache: bool = True) -> GlobalMagneticModel:
    cof = default_cof_path()
    if not cof.exists():
        try:
            download_wmm_cof(cof)
        except Exception:
            raise FileNotFoundError(
                f"Missing {cof}. Run: python -m ins_mag_nav.download_wmm"
            ) from None
    wmm = WMMModel(cof)
    grid = Path(__file__).resolve().parent / "data" / "global_wmm_grid_5deg.npz"
    if use_grid_cache and not grid.exists():
        try:
            build_global_grid(grid, step_deg=5.0)
        except Exception:
            grid = None
    return GlobalMagneticModel(wmm, grid if (use_grid_cache and grid.exists()) else None)


def build_global_grid(out: Path, step_deg: float = 5.0, alt_m: float = 0.0) -> None:
    """Precompute worldwide WMM grid (for fast global lookup)."""
    wmm = WMMModel()
    lats = np.arange(-90, 90 + step_deg * 0.5, step_deg)
    lons = np.arange(-180, 180, step_deg)
    field = np.zeros((len(lats), len(lons), 3))
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            field[i, j] = wmm.field_ned(la, lo, alt_m)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, lats=lats, lons=lons, field=field)
    print("Global WMM grid saved:", out, field.shape)
