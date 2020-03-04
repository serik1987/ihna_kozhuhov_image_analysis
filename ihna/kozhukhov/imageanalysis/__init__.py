"""
The module contains all functions that processes the optical imaging data
"""

try:
    from ._imageanalysis import *
except ImportError:
    raise ImportError("ihna/kozhukhov/imageanalysis/_imageanalysis.so not found")

from .imagingdata import ImagingData
from .imagingmap import ImagingMap
from .imagingsignal import ImagingSignal
