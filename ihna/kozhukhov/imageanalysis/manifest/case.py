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
                      'averaged_maps', 'auto', 'imported']

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
        print("load case")

    def save_case(self, parent_xml):
        print("save case")
        return None

    def set_properties(self, **kwargs):
        for property_name in self.property_names:
            if property_name in kwargs:
                self.__properties[property_name] = copy.deepcopy(kwargs[property_name])

    def __getitem__(self, key):
        if key in self.property_names:
            return self.__properties[key]
