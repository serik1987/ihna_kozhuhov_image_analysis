# -*- coding: utf-8

from .besselbox import BesselBox
from .butterbox import ButterBox
from .cheby1box import Cheby1Box
from .cheby2box import Cheby2Box
from .ellipbox import EllipBox
from .iirpeakbox import IirPeakBox
from .iirnotch import IirNotchBox

filters = {
    "butter": ButterBox,
    "cheby1": Cheby1Box,
    "cheby2": Cheby2Box,
    "ellip": EllipBox,
    "bessel": BesselBox,
    "iirnotch": IirNotchBox,
    "iirpeak": IirPeakBox
}
