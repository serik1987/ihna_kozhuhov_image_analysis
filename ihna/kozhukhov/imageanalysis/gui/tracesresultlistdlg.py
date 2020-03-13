# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis import ImagingSignal
from .resultlistdlg import ResultListDlg
from .signalviewerdlg import SignalViewerDlg


class TraceResultListDlg(ResultListDlg):

    def __init__(self, parent, case):
        super().__init__(parent, case)

    def _get_base_title(self):
        return "Trace list"

    def get_data_class(self):
        return ImagingSignal

    def _create_map_viewer_dlg(self, data):
        dlg = SignalViewerDlg(self, data, False)
        return dlg
