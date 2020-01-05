# -*- coding: utf-8

import xml.etree.ElementTree as ET
import numpy as np
from .roi import Roi


class ComplexRoi(Roi):
    """
    Represents a ROI that is a set of more elementary simple ROIs
    """

    __parent = None
    __roi_list = None

    def __init__(self, parent, name, subroi_list):
        """
        Initializes the subroi list.

        Arguments:
            parent - The ROI list
            name - name of the complex ROI
            subroi_list - list of all subroi names
        """
        self.__parent = parent
        self.set_name(name)
        self.set_subroi_list(subroi_list)

    def set_subroi_list(self, subroi_list):
        self.__roi_list = []
        for subroi in subroi_list:
            self.__roi_list.append(self.__parent[subroi])

    def get_left(self):
        """
        Returns the left border of the ROI
        """
        left = None
        for roi in self.__roi_list:
            if left is None or roi.get_left() < left:
                left = roi.get_left()
        return left

    def get_right(self):
        """
        Returns the right border of the ROI
        """
        right = None
        for roi in self.__roi_list:
            if right is None or roi.get_right() > right:
                right = roi.get_right()
        return right

    def get_top(self):
        """
        Returns the top border of the ROI
        """
        top = None
        for roi in self.__roi_list:
            if top is None or roi.get_top() < top:
                top = roi.get_top()
        return top

    def get_bottom(self):
        """
        Returns the bottom border of the ROI
        """
        bottom = None
        for roi in self.__roi_list:
            if bottom is None or roi.get_bottom() > bottom:
                bottom = roi.get_bottom()
        return bottom

    def get_width(self):
        """
        Returns the ROI width
        """
        x = np.zeros(self.get_right()) != 0
        for roi in self.__roi_list:
            for coordinates in roi.get_coordinate_list():
                x[coordinates[1]] = True
        return x.sum()

    def get_height(self):
        """
        Returns the ROI height
        """
        y = np.zeros(self.get_bottom()) != 0
        for roi in self.__roi_list:
            for coordinates in roi.get_coordinate_list():
                y[coordinates[0]] = True
        return y.sum()

    def get_coordinate_list(self):
        """
        Returns the list of all coordinates
        """
        m = np.zeros((self.get_bottom(), self.get_right())) != 0
        for roi in self.__roi_list:
            for coordinates in roi.get_coordinate_list():
                m[coordinates[0], coordinates[1]] = True
        lst = []
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j]:
                    lst.append([i, j])
        return lst

    def get_area(self):
        """
        Returns the total area of the ROI
        """
        m = np.zeros((self.get_bottom(), self.get_right())) != 0
        for roi in self.__roi_list:
            for coordinates in roi.get_coordinate_list():
                m[coordinates[0], coordinates[1]] = True
        return m.sum()

    def get_type(self):
        return "complex"

    def _save_details(self, xml):
        xml.attrib['type'] = "complex"
        for roi in self.__roi_list:
            element = ET.SubElement(xml, "subroi")
            element.text = roi.get_name()
            element.tail = "\n"

    def apply(self, data_map):
        """
        Applies ROI to the map

        Input arguments:
            data_map - numpy array
        """
        mask = np.zeros((data_map.shape[0], data_map.shape[1])) == 0
        for coordinate in self.get_coordinate_list():
            mask[coordinate[0], coordinate[1]] = False
        new_map = np.array(data_map, dtype=np.float)
        new_map[mask] = np.nan
        new_map = new_map[self.get_top():self.get_bottom(), self.get_left():self.get_right()]
        return new_map
