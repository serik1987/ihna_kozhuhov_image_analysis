# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.mapprocessing.correlation import ordinary_correlation, circular_correlation
from .twodatatonumberprocessor import TwoDataToNumberProcessor


class MapCorrelationProcessor(TwoDataToNumberProcessor):

    __is_circular_box = None

    def _get_default_minor_name(self):
        return "correlation"

    def _check_two_maps(self):
        first_feature = self._input_data.get_features()['type']
        second_feature = self._second_map.get_features()['type']
        h1 = self._input_data.get_harmonic()
        h2 = self._second_map.get_harmonic()
        if first_feature != second_feature:
            raise ValueError("Both maps shall have the same type")
        if h1 != h2:
            raise ValueError("Map harmonic mismatch")

    def _place_additional_options(self, parent):
        self.__is_circular_box = wx.CheckBox(parent, label="Compute circular cross-correlation")
        return self.__is_circular_box

    def _get_processor_title(self):
        return "Cross-correlation"

    def _process_maps(self):
        first_map = self._input_data
        second_map = self._second_map
        if self.__is_circular_box.GetValue():
            self._output_data = circular_correlation(first_map, second_map)
        else:
            self._output_data = ordinary_correlation(first_map, second_map)

    def _print_output_value(self):
        return "Correlation with " + self._second_map.get_full_name() + " is " + str(self._output_data)
