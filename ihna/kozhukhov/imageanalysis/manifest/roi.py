# -*- coding: utf-8


import xml.etree.ElementTree as ET


class Roi:
    """
    This is the base class that represents a certain ROI

    Derived classes: SimpleRoi, ComplexRoi
    """

    __name = None

    def __init__(self):
        """
        Creates new ROI and fills all its coordinates to None
        """
        raise NotImplementedError("ihna.kozhukhov.imageanalysis.manifest.Roi class is fully abstract. Use one "
                                  "of its derived classes")

    def get_left(self):
        raise NotImplementedError("get_left")

    def get_right(self):
        raise NotImplementedError("get_right")

    def get_top(self):
        raise NotImplementedError("get_top")

    def get_bottom(self):
        raise NotImplementedError("get_bottom")

    def get_width(self):
        raise NotImplementedError("get_width")

    def get_height(self):
        raise NotImplementedError("get_height")

    def get_area(self):
        raise NotImplementedError("get_area")

    def get_coordinate_list(self):
        raise NotImplementedError("get_coordinate_list")

    def get_name(self):
        """
        Returns the ROI name
        """
        if self.__name is None:
            raise AttributeError("Please, define the ROI name")
        else:
            return self.__name

    def set_name(self, value):
        """
        sels the ROI name
        """
        self.__name = str(value)

    def __str__(self):
        return "\
Name: {7}\n\
Left border: {0}\n\
Right border: {1}\n\
Top border: {2}\n\
Bottom border: {3}\n\
Width: {4}\n\
Height: {5}\n\
Area: {6}\n".format(self.get_left(), self.get_right(), self.get_top(), self.get_bottom(), self.get_width(),
                    self.get_height(), self.get_area(), self.get_name())

    def save(self):
        """
        Saves the ROI to the XML element

        Return: xml.etree.ElementTree.Element instance
        """
        element = ET.Element("roi", type="simple", name=self.get_name())
        element.text = "\n"
        self._save_details(element)
        return element

    def _save_details(self, xml):
        """
        Adds necessary children elements that specify certain ROI properties

        Arguments:
            xml - the <roi> element
        """
        raise NotImplementedError("_save_details")

    @staticmethod
    def load(xml):
        """
        Loads the ROI element

        Argument:
            xml - the <roi> element

        Return:
            instance of the ROI class
        """
        if xml.tag != "roi":
            raise ValueError("Please, specify the <roi> element")
        if xml.attrib['type'] == "simple":
            from .simpleroi import SimpleRoi
            roi = SimpleRoi()
        else:
            raise ValueError("Unknown or unsupported ROI type")
        roi.set_name(xml.attrib['name'])
        roi._load_details(xml)
        return roi

    def _load_details(self, xml):
        """
        Loads all ROI properties except name and type

        Arguments:
                xml - the <roi> element
        """
        raise NotImplementedError("_load_details")

    def get_type(self):
        """
        Returns a string containing the ROI type
        """
        raise NotImplementedError("get_type")

    def apply(self, data_map):
        """
        Applies ROI to the map and returns the adjusted ROI
        """
        raise NotImplementedError("apply")
