"""INS + magnetometer fusion (MEKF attitude + strapdown navigation)."""

from .fusion_session import FusionSession, ImuMagSample, InitialState, NavigationResult
from .loose_coupled_ekf import LooseCoupledEKF
from .global_mag_model import GlobalMagneticModel, load_global_model
from .magnetic_map import GeomagneticMap
from .wmm_pure import WMMModel
from .navigator import InsMagNavigator, NavState

__all__ = [
    "InsMagNavigator",
    "NavState",
    "FusionSession",
    "InitialState",
    "ImuMagSample",
    "NavigationResult",
    "LooseCoupledEKF",
    "GeomagneticMap",
    "GlobalMagneticModel",
    "load_global_model",
    "WMMModel",
]
