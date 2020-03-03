# -*- coding: utf-8

import os
import numpy as np
from ihna.kozhukhov.imageanalysis.accumulators import MapPlotter


class ImagingMap:
    """
    This is the base class that allows to receive, load and save the imaging ma
    """

    __features = None
    __data = None

    def __init__(self, arg, major_name=None):
        self.__features = {}
        if isinstance(arg, MapPlotter):
            self.__load_from_plotter(arg, major_name)
        else:
            self.__load_from_file(arg)

    def __load_from_plotter(self, plotter, major_name):
        self.__data = plotter.target_map
        self.__features["major_name"] = major_name
        self.__features["minor_name"] = "src"
        self.__features["type"] = "complex"
        self.__features["divide_by_average"] = str(plotter.divide_by_average)
        if plotter.preprocess_filter:
            self.__features["preprocess_filter"] = "{0} px".format(plotter.preprocess_filter_radius)
        isoline = plotter.isoline
        self.__features["isoline_class"] = isoline.__class__.__name__
        self.__features["analysis_epoch"] = "{0} - {1} cycles ({2} - {3} frames)".format(
            isoline.analysis_initial_cycle, isoline.analysis_final_cycle,
            isoline.analysis_initial_frame, isoline.analysis_final_frame
        )
        self.__features["isoline_epoch"] = "{0} - {1} cycles ({2} - {3} frames)".format(
            isoline.isoline_initial_cycle, isoline.isoline_final_cycle,
            isoline.isoline_initial_frame, isoline.isoline_final_frame
        )
        self.__features["harmonic"] = isoline.synchronization.harmonic
        self.__features["do_precise"] = str(isoline.synchronization.do_precise)
        self.__features["initial_phase"] = str(isoline.synchronization.initial_phase)
        self.__features["phase_increment"] = str(isoline.synchronization.phase_increment)
        train = isoline.synchronization.train
        self.__features["native_data"] = os.path.join(train.file_path, train.filename)
        self.__features["experiment_mode"] = train.experiment_mode
        self.__features["frame_shape"] = str(train.frame_shape)
        for src in train:
            break
        soft = src.soft
        self.__features["wavelegth"] = str(soft["wavelength"])
        self.__features["filter_width"] = str(soft["filter_width"])
        hard = src.isoi['hard']
        for property_name in [
            "camera_name", "camera_type", "resolution_x", "resolution_y", "pixel_size_x", "pixel_size_y",
            "ccd_aperture_x", "ccd_aperture_y", "integration_time", "interframe_time",
            "horizontal_hardware_binning", "vertical_hardware_binning", "hardware_gain", "hardware_offset",
            "ccd_size_x", "ccd_size_y", "dynamic_range", "optics_focal_length_top", "optics_focal_length_bottom"
        ]:
            self.__features[property_name] = str(hard[property_name])

    def __str__(self):
        S = ""
        for key, value in self.__features.items():
            S += "{0}: {1}\n".format(key, value)
        S += "Map data:\n"
        S += str(self.__data)
        return S

    def get_data(self):
        """
        Returns the map data
        """
        return self.__data

    def get_features(self):
        """
        Returns some scalar or text values assigned to the map
        """
        return self.__features

    def get_full_name(self):
        return "%s_%s" % (self.__features["major_name"], self.__features["minor_name"])

    def get_harmonic(self):
        return float(self.__features["harmonic"])
