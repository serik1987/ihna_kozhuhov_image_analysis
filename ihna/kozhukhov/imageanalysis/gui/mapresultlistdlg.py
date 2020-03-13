# -*- coding: utf-8

import wx
from .resultlistdlg import ResultListDlg


class MapResultListDlg(ResultListDlg):

    def __init__(self, parent, case):
        super().__init__(parent, case)

    def _get_base_title(self):
        return "Map list"
