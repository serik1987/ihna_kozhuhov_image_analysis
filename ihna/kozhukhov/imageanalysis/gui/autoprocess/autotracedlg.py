# -*- coding: utf-8

import wx
from .trainautoprocessdlg import TrainAutoprocessDlg


class AutotraceDlg(TrainAutoprocessDlg):

    def __init__(self, parent, animal_filter, autodecompress):
        super().__init__(parent, animal_filter, autodecompress, "Autotrace")
