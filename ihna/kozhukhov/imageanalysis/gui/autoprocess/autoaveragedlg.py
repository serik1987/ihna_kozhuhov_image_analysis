# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.gui.mapplotterdlg import MapPlotterDlg
from .trainautoprocessdlg import TrainAutoprocessDlg


class AutoaverageDlg(TrainAutoprocessDlg):

    def __init__(self, parent, animal_filter, autodecompress):
        super().__init__(parent, animal_filter, autodecompress, "Autoaverage")

    def _open_process_dlg(self, train):
        dlg = MapPlotterDlg(self._parent, train)
        self._train.close()
        return dlg
