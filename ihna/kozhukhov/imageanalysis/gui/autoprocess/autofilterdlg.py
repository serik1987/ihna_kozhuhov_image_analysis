# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.gui.mapfilterdlg.basicwindow import BasicWindow as MapFilterDlg
from .trainautoprocessdlg import TrainAutoprocessDlg


class AutofilterDlg(TrainAutoprocessDlg):

    def __init__(self, parent, animal_filter, autodecompress):
        super().__init__(parent, animal_filter, autodecompress, "Autofiltration")

    def _open_process_dlg(self, train):
        dlg = MapFilterDlg(self, train)
        self._train.close()
        return dlg
