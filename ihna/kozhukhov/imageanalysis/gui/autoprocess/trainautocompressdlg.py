# -*- coding: utf-8

import wx
from .autoprocessdlg import AutoprocessDlg


class TrainAutocompressDlg(AutoprocessDlg):

    def __init__(self, parent, animal_filter, title):
        super().__init__(parent, animal_filter, title)
