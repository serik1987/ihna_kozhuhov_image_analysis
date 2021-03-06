# -*- coding: utf-8

import wx
import numpy as np
from ihna.kozhukhov.imageanalysis import ImagingMap, PinwheelCenterList
from .resultlistdlg import ResultListDlg
from .complexmapviewerdlg import ComplexMapViewerDlg
from .amplitudemapviewerdlg import AmplitudeMapViewerDlg
from .phasemapviewer import PhaseMapViewer
from .pinwheelviewerdlg import PinwheelViewerDlg


class MapResultListDlg(ResultListDlg):

    def __init__(self, parent, case):
        super().__init__(parent, case)

    def _get_base_title(self):
        return "Map list"

    def get_data_class(self):
        return [ImagingMap, PinwheelCenterList]

    def _create_map_viewer_dlg(self, data):
        if isinstance(data, ImagingMap):
            if data.is_amplitude_map():
                return AmplitudeMapViewerDlg(self, data)
            elif data.is_phase_map():
                return PhaseMapViewer(self, data)
            elif data.is_complex_map():
                return ComplexMapViewerDlg(self, data)
            else:
                raise NotImplementedError("Can't work with this type of map")
        elif isinstance(data, PinwheelCenterList):
            return PinwheelViewerDlg(self, data)
        else:
            raise NotImplementedError("No viewer specified for such type of data")
