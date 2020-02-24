# -*- coding: utf-8

import xml.etree.ElementTree as ET


class Filter:
    """
    Selects animals or cases that are suitable

    Usage (Filter is fully abstract, use any of its derivatives, e.g,, AnimalFilter or CaseFilter):
    filter = AnimalFilter(animal_list) # creates new empty animal filter
    filter = AnimalFilter(animal_list, xml) # loads filter from the XML element

    data is any subscriptable object that represents list of the data items. A data item shall be accessed through
    the data by notation data[key] where key is key of the data item. All properties supported by the filter
    shall be assignable for reading through the data item using the following form:
    property_value = data[key][property_name]

    How to use the filter:
    for case in filter:
        do_something(case)
    Both AnimalFilter and CaseFilter iterate through cases, not animals!
    """

    __data = None
    __field_values = None

    def __init__(self, data, xml=None):
        self.__data = data
        self.__field_values = {}
        if xml is not None:
            self.__load_xml(xml)

    def __load_xml(self, xml):
        for property_element in xml.findall("field"):
            property_name = property_element.attrib['name']
            property_value = property_element.attrib['value']
            if property_name in self.get_filter_fields():
                self[property_name] = property_value

    def save(self, xml, tag_name):
        filter_element = ET.SubElement(xml, tag_name)
        filter_element.tail = "\n"
        filter_element.text = "\n"
        for property_name in self.get_filter_fields():
            property_value = self[property_name]
            if len(property_value) > 0:
                ET.SubElement(filter_element, "field", name=property_name, value=property_value).tail = "\n"

    def get_filter_fields(self):
        """
        Returns the list containing names of all fields available for the filtration
        """
        raise NotImplementedError("Attempt to use fully abstract class")

    def __getitem__(self, key):
        if key in self.get_filter_fields():
            if key in self.__field_values:
                return self.__field_values[key]
            else:
                return ""
        else:
            raise IndexError("The property '{0}' is not supported by the filter".format(key))

    def __setitem__(self, key, value):
        if key in self.get_filter_fields():
            self.__field_values[key] = value
        else:
            raise IndexError("The property '{0}' is not supported by the filter".format(key))

    def get_data(self):
        """
        Returns the animal list or case list that is used for filtration
        """
        return self.__data
