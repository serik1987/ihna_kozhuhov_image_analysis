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

    def _save_data(self, npz_filename):
        np.savez(npz_filename, data=self.__data)

    def _get_data_to_save(self):
        return {
            "DATA": self.__data
        }

    def _load_data(self, npz_filename):
        npz_file = np.load(npz_filename)
        self.__data = npz_file["data"]

    def _copy_data(self, data):
        self.__data = data

    def is_amplitude_map(self):
        return self.get_features()['type'] == "amplitude"

    def is_phase_map(self):
        return self.get_features()['type'] == "phase"

    def is_complex_map(self):
        return self.get_features()['type'] == "complex"

    def amplitude_map(self, minor_name="amplitude"):
        """
        Returns amplitude map from this complex map

        Parameters:
            minor_name - minor name of the new amplitude map
        """
        if not self.is_complex_map():
            raise AttributeError("This operation is available for complex maps only")
        if self.get_data() is None:
            self.load_data()
        features = self.get_features().copy()
        features['minor_name'] = minor_name
        features['type'] = "amplitude"
        features['original_map'] = self.get_full_name()
        features['is_main'] = "no"
        data = np.abs(self.get_data())
        new_map = ImagingMap(features, data)
        return new_map

    def phase_map(self, minor_name="phase"):
        """
        Returns a phase map in radians, range from -pi/H till pi/H

        Arguments:
            minor_name - minor name of new phase map
        """
        if not self.is_complex_map():
            raise AttributeError("This operation is available for complex map only")
        if self.get_data() is None:
            self.load_data()
        features = self.get_features().copy()
        features['minor_name'] = minor_name
        features['type'] = 'phase'
        features['original_map'] = self.get_full_name()
        features['is_main'] = 'no'
        h = self.get_harmonic()
        data = np.angle(self.get_data()) / h
        new_map = ImagingMap(features, data)
        return new_map

    @staticmethod
    def complex_map(amplitude_map, phase_map, minor_name="complex"):
        """
        Combines complex map from amplitude map and phase map

        Arguments:
            amplitude_map - an amplitude map
            phase_map - a phase map
            minor_name - minor name of new complex map
        """
        if not amplitude_map.is_amplitude_map():
            raise ValueError("The first argument must be an amplitude map")
        if not phase_map.is_phase_map():
            raise ValueError("The second argument myst be a phase map")
        features = phase_map.get_features().copy()
        features['minor_name'] = minor_name
        features['type'] = 'complex'
        features['original_map'] = amplitude_map.get_full_name() + "," + phase_map.get_full_name()
        features['is_main'] = 'no'
        h = phase_map.get_harmonic()
        A = amplitude_map.get_data()
        P = phase_map.get_data()
        C = A * np.exp(1j * h * P)
        new_map = ImagingMap(features, C)
        return new_map
