"""Download / refresh NOAA WMM coefficients. Run: python -m ins_mag_nav.download_wmm"""

from __future__ import annotations

import numpy as np

from .global_mag_model import build_global_grid, download_wmm_cof
from .wmm_pure import WMMModel, default_cof_path


def main() -> None:
    cof = default_cof_path()
    print("Downloading WMM.COF to", cof)
    try:
        download_wmm_cof(cof)
        print("OK:", cof)
    except Exception as e:
        print("Download failed (bundled file may already exist):", e)
    if cof.exists():
        w = WMMModel(cof)
        b = w.field_ned(39.9, 116.4, 50.0)
        print("Test Beijing field NED (nT):", b.round(2), " |B|=", round(float(np.linalg.norm(b)), 2))
    out = cof.parent / "global_wmm_grid_5deg.npz"
    print("Building global 5deg grid ->", out)
    build_global_grid(out)
    print("Done.")


if __name__ == "__main__":
    main()
