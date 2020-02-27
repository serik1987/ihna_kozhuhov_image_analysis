# -*- coding: utf-8

import wx
from .autoprocessdlg import AutoprocessDlg


class AutoFrameExtractDlg(AutoprocessDlg):

    def __init__(self, parent, animal_filter):
        super().__init__(parent, animal_filter, "Frame extraction")
