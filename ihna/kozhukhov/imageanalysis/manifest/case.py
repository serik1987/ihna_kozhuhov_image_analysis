# -*- coding: utf-8

import copy
import os.path
import xml.etree.ElementTree as ET


class Case:
    """
    The class represents some operations on the case list.

    The best way to create the case is to use cases_list. However, another options are possible:
    Case(xml) - loads the case from the XML element
    Case(valid_file) - import the case from the valid_file dictionary (returned by
    ihna.kozhukhov.imageanalysis.sourcefiles.get_file_info)

    The most of operations on a single cases are provided in CasesList module. Among those not provided are
    case[property] - to read the property value
    case[property] = value - to set the property value

    The following properties are available:
    case['pathname'] - path to all files related to the case
    case['filename'] - the filename (without the path)
    case['short_name'] - short name of the case
    case['long_name'] - long name of the case
    case['stimulation'] - the stimulus info
    case['additional_stimulation'] - additional stimulus info
    case['additional_information'] - additional information about the case
    case['native_data_files'] - list of all native files
    case['compressed_data_files'] - list of all compressed files
    case['roi'] - all ROI coordinates
    case['trace_files'] - list of all trace files
    case['averaged_maps'] - list of all averaged maps
    case['auto'] - if true, the case will be included in the autoprocess and autocompress

    In some cases, an efficient way may be provided by the case.set_properties(...)
    Function. For instance, in order to set the pathname, the filename and compressed_data_files
    simultaneously, use:
    case.set_properties(pathname = path, filename = file, compressed_data_files = [file1, file2, file3])
    """

    property_names = ['pathname', 'filename', 'short_name', 'long_name', 'stimulation', 'additional_stimulation',
                      'additional_information', 'native_data_files', 'compressed_data_files', 'roi', 'trace_files',
                      'averaged_maps', 'auto', 'imported', 'special_conditions']

    __properties = None

    def __init__(self, input_object):
        self.__properties = {}
        for property_name in self.property_names:
            self.__properties[property_name] = None
        self.__properties['auto'] = False
        self.__properties['imported'] = False
        if isinstance(input_object, dict):
            self.import_case(input_object)
        elif isinstance(input_object, ET.Element):
            self.load_case(input_object)
        else:
            raise ValueError("Unrecognized argument type for the case")

    def import_case(self, data):
        pathname, filename = os.path.split(data['filename'])
        filetype = data['filetype']
        train_files = [filename]
        if filetype == "compressed" or filetype == "stream":
            for tail_file in data['tail_files']:
                train_files.append(os.path.split(tail_file)[1])
        if filetype == "analysis":
            self.set_properties(pathname=pathname, filename=filename, averaged_maps=[filename], imported=True)
        elif filetype == "compressed":
            self.set_properties(pathname=pathname, filename=filename, compressed_data_files=train_files, imported=True)
        elif filetype == "stream":
            self.set_properties(pathname=pathname, filename=filename, native_data_files=train_files, imported=True)
        else:
            self.set_properties(pathname=pathname, filename=filename, imported=True)

    def load_case(self, xml):
        if xml.attrib['autoprocess'] == "Y":
            autoprocess = True
        else:
            autoprocess = False
        self.set_properties(
            filename=xml.attrib['filename'],
            short_name=xml.attrib['short_name'],
            long_name=xml.attrib['long_name'],
            stimulation=xml.attrib['stimulation'],
            additional_stimulation=xml.attrib['additional_stimulation'],
            special_conditions=xml.attrib['special_conditions'],
            additional_information=xml.attrib['additional_information'],
            auto=autoprocess
        )
        if xml.find('native-files') is not None:
            self['native_data_files'] = self.__read_file_list(xml.find('native-files'), 'native-file')
        if xml.find('compressed-files') is not None:
            self['compressed_data_files'] = self.__read_file_list(xml.find('compressed-files'), 'compressed-file')
        if xml.find('trace-files') is not None:
            self['trace_files'] = self.__read_file_list(xml.find('trace-files'), 'trace-file')
        if xml.find('averaged-maps') is not None:
            self['averaged_maps'] = self.__read_file_list(xml.find('averaged-maps'), 'averaged-map')
        if xml.find('roi') is not None:
            self.__load_roi(xml.find('roi'))

    def __read_file_list(self, xml, element_name):
        filelist = []
        for file_element in xml.findall(element_name):
            filelist.append(file_element.text)
        return filelist

    def __load_roi(self, xml):
        self['roi'] = []

    def save_case(self, parent_xml):
        if self['imported']:
            return
        if self['auto']:
            autoprocess_line = "Y"
        else:
            autoprocess_line = "N"
        root = ET.SubElement(parent_xml, "case", {
            "filename": self['filename'],
            "short_name": self['short_name'],
            "long_name": self['long_name'],
            "stimulation": self['stimulation'],
            "additional_stimulation": self['additional_stimulation'],
            "special_conditions": self['special_conditions'],
            "additional_information": self['additional_information'],
            "autoprocess": autoprocess_line
        })
        root.tail = "\n"
        root.text = "\n"
        if self['native_data_files'] is not None:
            self.__add_file_list(root, "native-files", "native-file", self['native_data_files'])
        if self['compressed_data_files'] is not None:
            self.__add_file_list(root, "compressed-files", "compressed-file", self['compressed_data_files'])
        if self['trace_files'] is not None:
            self.__add_file_list(root, "trace-files", "trace-file", self['trace_files'])
        if self['averaged_maps'] is not None:
            self.__add_file_list(root, "averaged-maps", "averaged-map", self['averaged_maps'])
        if self['roi'] is not None:
            self.__add_roi(root)

    def __add_file_list(self, root, list_name, list_element_name, filelist):
        list_element = ET.SubElement(root, list_name)
        list_element.text = "\n"
        list_element.tail = "\n"
        for filename in filelist:
            filename_element = ET.SubElement(list_element, list_element_name)
            filename_element.text = filename
            filename_element.tail = "\n"

    def __add_roi(self, root):
        roi_list_element = ET.SubElement(root, "roi")
        roi_list_element.tail = "\n"

    def set_properties(self, **kwargs):
        for property_name in self.property_names:
            if property_name in kwargs:
                self.__properties[property_name] = copy.deepcopy(kwargs[property_name])

    def __getitem__(self, key):
        if key in self.property_names:
            return self.__properties[key]
        else:
            raise IndexError("Subscript index doesn't refer to the valid property name")

    def __setitem__(self, key, value):
        if key in self.property_names:
            self.__properties[key] = value
        else:
            raise IndexError("Subscript index doesn't refer to the valid property name")

    def __str__(self):
        s = ""
        s += "Pathname: %s\n" % (self['pathname'])
        s += "Filename: %s\n" % (self['filename'])
        s += "Short name: %s\n" % (self['short_name'])
        s += "Long name: %s\n" % (self['long_name'])
        s += "Stimulation: %s\n" % (self['stimulation'])
        s += "Additional stimulation: %s\n" % (self['additional_stimulation'])
        s += "Special conditions: %s\n" % (self['special_conditions'])
        s += "Additional information: %s\n" % (self['additional_information'])
        s += "Native data files: {0}\n".format(self['native_data_files'])
        s += "Compressed data files: {0}\n".format(self['compressed_data_files'])
        s += "ROI information:\n{0}\n".format(self['roi'])
        s += "Trace files: {0}\n".format(self['trace_files'])
        s += "Averaged_maps: {0}\n".format(self['averaged_maps'])
        if self['auto']:
            s += "Autoprocess on\n"
        else:
            s += "Autoprocess off\n"
        if self['imported']:
            s += "Case info is not presented"
        else:
            s += "Case info is presented"
        return s

    def get_discarded_list(self):
        discarded_list = []
        short_list = []
        if self['native_data_files'] is not None:
            short_list.extend(self['native_data_files'])
        if self['compressed_data_files'] is not None:
            short_list.extend(self['compressed_data_files'])
        if self['trace_files'] is not None:
            short_list.extend(self['trace_files'])
        if self['averaged_maps'] is not None:
            short_list.extend(self['averaged_maps'])
        for filename in short_list:
            fullname = os.path.join(self['pathname'], filename)
            discarded_list.append(fullname)
        return discarded_list
