# -*- coding: utf-8

import wx
import numpy as np
from ihna.kozhukhov.imageanalysis import ImagingMap
from .resultlistdlg import ResultListDlg
from .complexmapviewerdlg import ComplexMapViewerDlg
from .amplitudemapviewerdlg import AmplitudeMapViewerDlg


class MapResultListDlg(ResultListDlg):

    def __init__(self, parent, case):
        super().__init__(parent, case)

    def _get_base_title(self):
        return "Map list"

    def get_data_class(self):
        return ImagingMap

    def _create_map_viewer_dlg(self, data: ImagingMap):
        if data.get_data().dtype == np.complex128:
            return ComplexMapViewerDlg(self, data)
        else:
            if "map_type" in data.get_features().keys():
                raise NotImplementedError("Can't work with map_type")
            else:
                return AmplitudeMapViewerDlg(self, data)
