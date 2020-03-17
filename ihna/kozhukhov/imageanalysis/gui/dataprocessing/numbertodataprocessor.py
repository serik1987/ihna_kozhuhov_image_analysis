# -*- coding: utf-8

import wx
from .DataProcessDlg import DataProcessDlg


class NumberToDataProcessor(DataProcessDlg):

    def _get_processor_title(self):
        return "Sample number-to-data processor"

    def _check_input_data(self):
        pass

    def _place_general_options(self, parent):
        general_options = wx.BoxSizer(wx.VERTICAL)

        major_name_box = self._place_major_name(parent)
        general_options.Add(major_name_box, 0, wx.EXPAND | wx.BOTTOM, 5)

        minor_name_box = self._place_output_file_box(parent)
        general_options.Add(minor_name_box, 0, wx.EXPAND | wx.BOTTOM, 5)

        save_details_box = self._place_save_details(parent)
        general_options.Add(save_details_box, 0, wx.EXPAND)

        return general_options
