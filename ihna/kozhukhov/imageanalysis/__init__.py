# -*- coding: utf-8
"""
This package contains all scripts for imageanalysis.

Particularly:
gui - working in graphical interface with wxPython

manifest - dealing with electronic lab journal
sourcefiles - dealing with native data files
compression - turning native data into compressed mode and returning them from the compressed mode
synchronization, isolines, tracereading, accumulators - responsible for conversion of the native data into
averaged maps, oscillatory maps, averaged and individual traces
mapprocessing - processing of the averaged and oscillatory maps
traceprocessing - processing of the result traces

Besides these packages the package contains three main classes
PinwheelCenterList - represents list of all pinwheel centers for the map
ImagingMap - represents averaged and oscillatory maps
ImagingSignal - represents averaged traces
ImagingData - base class for all other imaging data
"""

try:
    from ._imageanalysis import *
except ImportError:
    raise ImportError("ihna/kozhukhov/imageanalysis/_imageanalysis.so not found")

from .imagingdata import ImagingData
from .imagingmap import ImagingMap
from .imagingsignal import ImagingSignal
from .pinwheelcenterlist import PinwheelCenterList

imaging_data_classes = {
    "ImagingMap": ImagingMap,
    "ImagingSignal": ImagingSignal,
    "PinwheelCenterList": PinwheelCenterList
}
