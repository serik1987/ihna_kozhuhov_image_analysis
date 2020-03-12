# -*- coding: utf-8

import wx
from .resultlistdlg import ResultListDlg


class TraceResultListDlg(ResultListDlg):

    def __init__(self, parent, case):
        super().__init__(parent, case)
