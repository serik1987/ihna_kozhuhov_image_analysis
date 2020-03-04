# -*- coding: utf-8

import os
import xml.etree.ElementTree as ET


class ImagingData:
    """
    This is the base class that shows processing results of the optical imaging data.

    Don't use this abstract class. Use any of its derived classes
    """

    __features = None

    def __init__(self, arg, major_name=None):
        """
        Ways to create new map:
            imaging_map = ImagingMap(plotter, major_name)
            where: plotter is MapPlotter instance where all the data have been accumulated
            major_name - a string containing animal name, case short name, name prefix and name postfix
            'src' will be added as a minor name
        """
        self.__features = {}
        if isinstance(arg, ET.Element):
            self.__load_from_file(arg)
        else:
            self.__load_from_plotter(arg, major_name)

    def __load_from_plotter(self, plotter, major_name):
        self.__features["major_name"] = major_name
        self.__features["minor_name"] = "src"
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
        self._load_imaging_data_from_plotter(plotter)

    def get_features(self):
        """
        Returns some scalar or text values assigned to the map. All features will be returned as
        string values. The following ones are used in the map processing:
        'harmonic' - 1.0 for direction or retinotopic maps, 2.0 for orientation maps
        """
        return self.__features

    def _load_imaging_data_from_plotter(self, plotter):
        """
        Loading data-specific properties from the map loader
        """
        raise NotImplementedError("_load_imaging_data_from_plotter")

    def get_data(self):
        """
        Returns the data-specific data as tuples and numpy array. The format depends on a certain imaging data
        """
        raise NotImplementedError("get_data")

    def __str__(self):
        S = ""
        for key, value in self.__features.items():
            S += "{0}: {1}\n".format(key, value)
        S += "Map data:\n"
        S += str(self.get_data())
        return S

    def get_full_name(self):
        """
        Map full name that contains its major and minor names
        """
        return "%s_%s" % (self.__features["major_name"], self.__features["minor_name"])
