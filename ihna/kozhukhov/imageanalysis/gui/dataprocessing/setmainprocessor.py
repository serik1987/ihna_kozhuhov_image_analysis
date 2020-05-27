# -*- coding: utf-8

import numpy as np
import wx
from .datatonumberprocessor import DataToNumberProcessor
from ihna.kozhukhov.imageanalysis import ImagingMap


class SetMainProcessor(DataToNumberProcessor):

    __is_main_box = None

    def _check_input_data(self):
        if isinstance(self._input_data, ImagingMap) and self._input_data.get_data().dtype == np.complex:
            return
        raise ValueError("This processor requires complex maps only")

    def _get_processor_title(self):
        return "Map average value"

    def _get_default_minor_name(self):
        return "is_main"

    def _place_additional_options(self, parent):
        self.__is_main_box = wx.CheckBox(parent, label="Use this map for ROI selection")
        input_data = self._input_data
        if 'is_main' in input_data.get_features() and input_data.get_features()['is_main'] == "yes":
            self.__is_main_box.SetValue(True)
        else:
            self.__is_main_box.SetValue(False)

        return self.__is_main_box

    def _process(self):
        if self.__is_main_box.IsChecked():
            self._output_data = "yes"
        else:
            self._output_data = "no"

    def _print_output_value(self):
        if self._output_data == "yes":
            return "This map will be used for ROI selection"
        else:
            return "This map will not be used for ROI selection"
