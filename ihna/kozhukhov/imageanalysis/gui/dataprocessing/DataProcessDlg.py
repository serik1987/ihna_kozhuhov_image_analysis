# -*- coding: utf-8

import wx


class DataProcessDlg(wx.Dialog):

    _input_data = None
    _parent = None
    __output_data = None

    def __init__(self, parent, input_data):
        self._input_data = input_data
        self._parent = parent
        self._check_input_data()
        super().__init__(parent,
                         title="%s: %s" % (self._get_processor_title(), input_data.get_full_name()),
                         size=(800, 500))

        self.Centre()

    def _get_processor_title(self):
        return "Sample processor"

    def _check_input_data(self):
        raise NotImplementedError("Unable to use abstract data processor")
