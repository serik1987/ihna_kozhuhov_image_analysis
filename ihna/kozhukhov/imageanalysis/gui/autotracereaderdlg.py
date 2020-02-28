# -*- coding: utf-8

import wx
from .accumulatordlg import AccumulatorDlg


class AutotraceReaderDlg(AccumulatorDlg):

    __roi_box = None

    def __init__(self, parent, train):
        super().__init__(parent, train, "Autoreading")

    def _create_accumulator_box(self, parent):
        panel = wx.StaticBoxSizer(wx.VERTICAL, parent, label="ROI information")
        layout = wx.BoxSizer(wx.HORIZONTAL)

        roi_caption = wx.StaticText(parent, label="ROI name")
        layout.Add(roi_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        self.__roi_box = wx.TextCtrl(parent)
        layout.Add(self.__roi_box, 1)

        panel.Add(layout, 1, wx.ALL | wx.EXPAND, 5)
        return panel
