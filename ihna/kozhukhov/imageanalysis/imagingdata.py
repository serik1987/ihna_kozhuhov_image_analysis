# -*- coding: utf-8

import os
import json
import xml.etree.ElementTree as ET
import scipy.io


class ImagingData:
    """
    This is the base class that shows processing results of the optical imaging data.

    Don't use this abstract class. Use any of its derived classes
    """

    __features = None
    __filename = None

    def __init__(self, arg, major_name=None):
        """
        Ways to create new map:
            imaging_map = ImagingMap(plotter, major_name)
            where: plotter is MapPlotter instance where all the data have been accumulated
            major_name - a string containing animal name, case short name, name prefix and name postfix
            'src' will be added as a minor name by default.

            imaging_map = ImagingMap(json_file)
            will load the imaging element from the JSON file. This file will be created when you use save_npz function
            Actually, this option will load previously saved data
        """
        self.__features = {}
        if isinstance(arg, str):
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

    def __load_from_file(self, filename):
        json_filename = filename
        npz_filename = filename[:-4] + "npz"
        json_file = open(json_filename, "r")
        json_data = json_file.read()
        self.__features = json.loads(json_data)
        json_file.close()
        self.__filename = npz_filename

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

    def save_npz(self, folder_name):
        """
        Saves the data to the NPZ and JSON files

        Arguments:
            folder_name - folder where all files will be located. Names of these files will be the same as names
            of the animal
        """
        json_features = json.JSONEncoder().encode(self.__features)
        json_filename = os.path.join(folder_name, self.get_full_name() + ".json")
        npz_filename = os.path.join(folder_name, self.get_full_name() + ".npz")
        json_file = open(json_filename, "w")
        json_file.write(json_features)
        json_file.close()
        self._save_data(npz_filename)
        self.__filename = npz_filename

    def save_mat(self, folder_name):
        """
        Saves the data to MAT file

         Arguments:
            folder_name - folder where all files will be located. Names of these files will be the same as names
            of the animal
        """
        output = self.__features
        output.update(self._get_data_to_save())
        filename = os.path.join(folder_name, self.get_full_name() + ".mat")
        scipy.io.savemat(filename, output)

    def _save_data(self, npz_filename):
        """
        Creates the NPZ file itself

        Arguments:
            npz_filename - full name of the NPZ file

        The method shall consider the case when NPZ file is not loaded.
        """
        raise NotImplementedError("self._save_data " + npz_filename)

    def load_data(self):
        """
        All the ImagingData stores in two files: json and npz. json file stores small scalar values while
        npz file stores large numpy arrays. when calling ImagingData(json_file) you load only small scalar
        data from the json file. In order to load large numpy arrays from the npz file please, call this
        method
        """
        self._load_data(self.__filename)

    def _load_data(self, npz_filename):
        """
        Loads the data from NPZ file itself assuming that all data features have already been loaded
        """
        raise NotImplementedError("self._load_data " + npz_filename)

    def _get_data_to_save(self):
        raise NotImplementedError("_get_data_to_save")

    def get_filename(self):
        """
        Full name of the NPZ file where the data were saved or None when the data were not associated with any file
        """
        return self.__filename
