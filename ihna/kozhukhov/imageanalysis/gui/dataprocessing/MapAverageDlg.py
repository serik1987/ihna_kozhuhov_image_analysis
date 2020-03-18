# -*-  coding: utf-8

import wx
from numpy import mean, pi, isnan
from scipy.stats import circmean
from ihna.kozhukhov.imageanalysis import ImagingMap
from .datatonumberprocessor import DataToNumberProcessor


class MapAverageDlg(DataToNumberProcessor):

    __compute_circular_average = None

    def _get_processor_title(self):
        return "Map average value"

    def _check_input_data(self):
        return self._check_imaging_map(True)

    def _get_default_minor_name(self):
        return "average"

    def _place_additional_options(self, parent):
        self.__compute_circular_average = wx.CheckBox(parent, label="Compute a circular average")

        return self.__compute_circular_average

    def _process(self):
        input_data = self._input_data.get_data()
        if self.__compute_circular_average.IsChecked():
            harmonic = self._input_data.get_harmonic()
            value = circmean(input_data, high=pi/harmonic, low=-pi/harmonic, nan_policy="omit")
        else:
            input_vector = input_data.reshape(input_data.size)
            input_vector[isnan(input_vector)] = []
            value = mean(input_vector)
        self._output_data = value

    def _print_output_value(self):
        return "Average value: " + str(self._output_data)
