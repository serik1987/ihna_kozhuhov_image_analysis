# -*- coding: utf-8

import xml.etree.ElementTree as ET
from .roi import Roi


class SimpleRoi(Roi):
    """
    Represents a simple roi that have rectangular form
    """

    __left = None
    __right = None
    __bottom = None
    __top = None

    def __init__(self):
        pass

    def get_left(self):
        """
        Returns the left border of the ROI
        """
        if self.__left is None:
            raise AttributeError("Please, set the left border by application of set_left")
        else:
            return self.__left

    def set_left(self, value):
        """
        Sets the left border of the ROI
        """
        x = int(value)
        if (self.__right is None or self.__right > x) and x >= 0:
            self.__left = x
        else:
            raise ValueError("Value of the left border is not correct")

    def get_right(self):
        """
        Returns the right border of the ROI
        """
        if self.__right is None:
            raise AttributeError("Please, set the right border by application of set_right")
        else:
            return self.__right

    def set_right(self, value):
        """
        Sets the right border of the ROI
        """
        x = int(value)
        if (self.__left is None or self.__left < x) and x > 0:
            self.__right = x
        else:
            raise ValueError("Value of the right border is not correct")

    def get_top(self):
        """
        Returns the top border of the ROI
        """
        if self.__top is None:
            raise AttributeError("Please, set the top border by application of set_top")
        else:
            return self.__top

    def set_top(self, value):
        """
        Sets the top border of the ROI
        """
        x = int(value)
        if (self.__bottom is None or self.__bottom > x) and x >= 0:
            self.__top = x
        else:
            raise ValueError("Value of the top border is not correct")

    def get_bottom(self):
        """
        Returns the bottom border of the ROI
        """
        if self.__bottom is None:
            raise AttributeError("Please, define the bottom border of the ROI")
        else:
            return self.__bottom

    def set_bottom(self, value):
        """
        Sets the bottom border of the ROI
        """
        x = int(value)
        if (self.__top is None or self.__top < x) and x > 0:
            self.__bottom = x
        else:
            raise ValueError("Value of the ROI bottom border is not correct")

    def get_width(self):
        """
        Returns the ROI width
        """
        return self.get_right() - self.get_left()

    def get_height(self):
        """
        Returns the ROI height
        """
        return self.get_bottom() - self.get_top()

    def get_area(self):
        """
        Returns the ROI area
        """
        return self.get_width() * self.get_height()

    def get_coordinate_list(self):
        """
        Returns list of coordinates of all pixels included in the ROI like:
        [[i0, j0],
        [i1, j1],
        [i2, j2],
        ...
        ]
        where i relates to the row and j relates to the column
        """
        coordinate_list = []
        for x in range(self.get_left(), self.get_right()):
            for y in range(self.get_top(), self.get_bottom()):
                coordinate_list.append((y, x))
        return coordinate_list

    def _save_details(self, xml):
        left = ET.SubElement(xml, "left")
        left.text = str(self.get_left())
        left.tail = "\n"
        right = ET.SubElement(xml, "right")
        right.text = str(self.get_right())
        right.tail = "\n"
        top = ET.SubElement(xml, "top")
        top.text = str(self.get_top())
        top.tail = "\n"
        bottom = ET.SubElement(xml, "bottom")
        bottom.text = str(self.get_bottom())
        bottom.tail = "\n"

    def _load_details(self, xml):
        self.set_left(xml.find("left").text)
        self.set_right(xml.find("right").text)
        self.set_top(xml.find("top").text)
        self.set_bottom(xml.find("bottom").text)

    def get_type(self):
        return "simple"

    def apply(self, data_map):
        """
        Applies ROI to the map and returns the adjusted ROI
        """
        print("entry")
        return data_map[self.get_top():self.get_bottom(), self.get_left():self.get_right()]
