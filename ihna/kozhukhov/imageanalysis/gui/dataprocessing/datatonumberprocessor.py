# -*- coding: utf-8
from abc import ABC

import wx
from ihna.kozhukhov.imageanalysis import ImagingData
from .DataProcessDlg import DataProcessDlg


class DataToNumberProcessor(DataProcessDlg):

    def _check_input_data(self):
        if not isinstance(self._input_data, ImagingData):
            raise ValueError("This processor requires a single imaging data as an input")

    def _place_general_options(self, parent):
        return self._place_value_save_details(parent)

    def _print_output_value(self):
        """
        Returns the string containing body of the message with information about output value
        """
        raise NotImplementedError("DataToNumberProcessor._print_output_value()")

    def _save_processed_data(self):
        if self._output_data is None:
            raise NotImplementedError("The _process function shall save the value received to _output_data field")

        dlg = wx.MessageDialog(self, self._print_output_value(), self._get_processor_title(),
                               wx.OK | wx.CENTRE | wx.ICON_INFORMATION)
        dlg.ShowModal()

        if self.is_add_to_map_features():
            feature_name = self.get_value_key()
            feature_value = str(self._output_data)
            self._input_data.get_features()[feature_name] = feature_value
            folder_name = self._considering_case['pathname']
            self._input_data.save_npz(folder_name)

    def _get_result_viewer(self):
        pass
