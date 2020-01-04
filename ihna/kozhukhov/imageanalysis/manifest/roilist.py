# -*- coding: utf-8

import xml.etree.ElementTree as ET


class RoiList:
    """
    Specifies the ROI list

    Operations on the ROI list:
        list.add(roi) - Adds the ROI to the list
        list[name] - access the ROI by its name (must be a string)
        del list[name] - deletes the ROI with a certain index
        print(list) - prints the ROI list
        for roi in list:
            do_something(roi)
        The list is iterable!
        len(list) - total number of ROI in the list
        list.save() - saves all ROIs to a single XML element
        list.load(xml) - loads all ROIs from the XML element
    """

    __list = None

    def __init__(self):
        """
        Creates an empty ROI list
        """
        self.__list = {}

    def __str__(self):
        s = "name                \ttype      \tleft\tright\ttop\tbottom\twidth\theight\tarea\n"
        for name, roi in self.__list.items():
            s += "{0:20}\t{1:10}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(
                roi.get_name(),
                roi.get_type(),
                roi.get_left(),
                roi.get_right(),
                roi.get_top(),
                roi.get_bottom(),
                roi.get_width(),
                roi.get_height(),
                roi.get_area()
            )
        return s

    def add(self, roi):
        """
        Adds the ROI to the list
        """
        self.__list[roi.get_name()] = roi

    def __getitem__(self, name):
        return self.__list[name]

    def __delitem__(self, name):
        del self.__list[name]

    def __iter__(self):
        return iter(self.__list.values())

    def __len__(self):
        return len(self.__list)

    def save(self):
        """
        Saves all ROI in the list to a single XML element

        Return:
            XML element containing all information about ROIs
        """
        element = ET.Element("roi-list")
        element.text = "\n"
        for roi in self:
            sub_element = roi.save()
            sub_element.tail = "\n"
            element.append(sub_element)
        return element

    def load(self, xml):
        """
        Loads all ROI from the XML element and adds them to the list
        """
        self.__list = {}
        if xml.tag != "roi-list":
            raise ValueError("The value is not an <roi-list> element")
        from .roi import Roi
        complex_roi_list = []
        for roi_element in xml.findall("roi"):
            if roi_element.attrib["type"] == "complex":
                complex_roi_list.append(roi_element)
            else:
                roi = Roi.load(roi_element)
                self.add(roi)
        from .complexroi import ComplexRoi
        for complex_roi_element in complex_roi_list:
            roi = ComplexRoi(self, complex_roi_element.attrib['name'], [])
            subroi_names = []
            for subroi_element in complex_roi_element.findall("subroi"):
                subroi_names.append(subroi_element.text)
            roi.set_subroi_list(subroi_names)
            self.add(roi)
