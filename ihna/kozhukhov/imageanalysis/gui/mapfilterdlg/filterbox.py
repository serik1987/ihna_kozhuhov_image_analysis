# -*- coding: utf-8

import wx


class FilterBox(wx.BoxSizer):

    _dlg = None
    _parent = None

    __sample_widget = None

    def __init__(self, dlg, parent):
        super().__init__(wx.VERTICAL)
        self._dlg = dlg
        self._parent = parent
        self._create_widgets()

    def _get_filter_name(self):
        raise NotImplementedError("_get_filter_name")

    def _create_widgets(self):
        sample_widget = wx.StaticText(self._parent, label=self._get_filter_name())
        self.Add(sample_widget, 0, wx.EXPAND)
        self.__sample_widget = sample_widget

    def show(self):
        self.__sample_widget.Show(True)
        self.Layout()

    def hide(self):
        self.__sample_widget.Show(False)
        self.Layout()
