# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis import ImagingData
from .DataProcessDlg import DataProcessDlg


class DataToDataProcessor(DataProcessDlg):

    def _get_processor_title(self):
        return "Sample data-to-data processor"

    def _check_input_data(self):
        if not isinstance(self._input_data, ImagingData):
            raise ValueError("Please, select an appropriate imaging data for processing")

    def _place_general_options(self, parent):
        general_options = wx.BoxSizer(wx.VERTICAL)

        output_file_box = self._place_output_file_box(parent)
        general_options.Add(output_file_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        save_details = self._place_save_details(parent)
        general_options.Add(save_details, 0, wx.EXPAND)

        return general_options

    def _save_processed_data(self):
        if self._output_data is None:
            raise AttributeError("The data-to-data processor shall put the output map into _output_data field")
        self._save_output_data()
