# -*- coding: utf-8

import os
import numpy as np
from .imagingdata import ImagingData
from ihna.kozhukhov.imageanalysis.accumulators import MapPlotter


class ImagingMap(ImagingData):
    """
    This is the base class that allows to work with maps, load and save them, create them from the map
    plotter
    """

    __data = None

    def _load_imaging_data_from_plotter(self, plotter):
        self.get_features()["type"] = "complex"
        self.get_features()["divide_by_average"] = str(plotter.divide_by_average)
        if plotter.preprocess_filter:
            self.get_features()["preprocess_filter"] = "{0} px".format(plotter.preprocess_filter_radius)
        self.__data = plotter.target_map

    def get_data(self):
        """
        Returns the map data as 2D numpy matrix
        """
        return self.__data

    def get_harmonic(self):
        """
        Returns 1.0 for directional and retinotopic maps, 2.0 for orientation maps
        """
        return float(self.get_features()["harmonic"])
