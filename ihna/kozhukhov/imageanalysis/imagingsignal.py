# -*- coding: utf-8

from .imagingdata import ImagingData


class ImagingSignal(ImagingData):

    __times = None
    __values = None

    def _load_imaging_data_from_plotter(self, reader):
        self.__times = reader.times
        self.__values = reader.averaged_signal

    def get_data(self):
        """
        Returns tuple containing all times and all values assigned to the graph
        """
        return self.__times, self.__values

    def get_times(self):
        """
        Returns vector containing times
        """
        return self.__times

    def get_values(self):
        """
        Returns vector containing values from the imaging signal
        """
        return self.__values
