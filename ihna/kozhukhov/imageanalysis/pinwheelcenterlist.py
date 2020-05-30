# -*- coding: utf-8

import numpy as np
from .imagingdata import ImagingData


class PinwheelCenterList(ImagingData):

    __center_list = None
    __width = None
    __height = None

    def get_data(self):
        """
        Returns tuple containing center list, map width and map height.
        """
        return self.__center_list, self.__width, self.__height

    def set_map_size(self, width, height):
        """
        Adjusts size of the original map in pixels

        Arguments:
             width - map width
             height - map height
        """
        w = int(width)
        h = int(height)
        if w < 0 or h < 0:
            raise ValueError("Wrong map dimensions")
        self.__width = w
        self.__height = h

    def getMapWidth(self):
        """
        Returns map width in pixels
        """
        return self.__width

    def getMapHeight(self):
        """
        Returns map height in pixels
        """
        return self.__height

    def getCenterList(self):
        """
        Returns list containing all pinwheel coordinates as tuples
        (Abscissa is first item in the tuple, ordinate is the second one)
        """
        return self.__center_list

    def _save_data(self, npz_filename):
        packed_data = np.array(self.__center_list)
        np.savez(npz_filename, data=packed_data, width=self.__width, height=self.__height)

    def _load_data(self, npz_filename):
        npz_file = np.load(npz_filename)
        packed_data = npz_file['data']
        self.__load_from_matrix(packed_data)
        self.__width = npz_file['width']
        self.__height = npz_file['height']

    def __load_from_matrix(self, matrix):
        self.__center_list = [tuple(point) for point in list(matrix)]

    def _get_data_to_save(self):
        return {
            "DATA": np.array(self.__center_list),
            "MAP_WIDTH": self.__width,
            "MAP_HEIGHT": self.__height
        }

    def _copy_data(self, data):
        self.__center_list = data[0].copy()
        self.__width = int(data[1])
        self.__height = int(data[2])

    def _load_imaging_data_from_plotter(self, plotter):
        raise RuntimeError("These data can't be revealed from the plotter")
