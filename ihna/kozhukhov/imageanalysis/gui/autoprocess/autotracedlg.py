# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.gui.autotracereaderdlg import AutotraceReaderDlg
from .trainautoprocessdlg import TrainAutoprocessDlg


class AutotraceDlg(TrainAutoprocessDlg):

    def __init__(self, parent, animal_filter, autodecompress):
        super().__init__(parent, animal_filter, autodecompress, "Autotrace")

    def _open_process_dlg(self, train):
        dlg = AutotraceReaderDlg(self._parent, self._train)
        self._train.close()
        return dlg
