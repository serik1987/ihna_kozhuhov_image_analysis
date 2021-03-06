# -*- coding: utf-8

import copy
import os.path
import xml.etree.ElementTree as ET
from ihna.kozhukhov.imageanalysis import imaging_data_classes, ImagingSignal, ImagingMap
import ihna.kozhukhov.imageanalysis.sourcefiles as sfiles
from ihna.kozhukhov.imageanalysis.tracereading import Traces
from .roilist import RoiList


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
    case['special_conditions'] - special conditions
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

    __animal_name = None
    __case_list = None
    __properties = None
    __roi_list = None
    __traces_list = None
    __result_list = None

    def __init__(self, input_object, pathname=None, animal_name=None, case_list=None):
        self.__properties = {}
        for property_name in self.property_names:
            self.__properties[property_name] = None
        self.__properties['auto'] = False
        self.__properties['imported'] = False
        self.__roi_list = RoiList()
        self.__traces_list = []
        self.__result_list = []
        if isinstance(input_object, dict):
            self.import_case(input_object)
        elif isinstance(input_object, ET.Element):
            self.load_case(input_object, pathname)
        else:
            raise ValueError("Unrecognized argument type for the case")
        self.__animal_name = animal_name
        self.__case_list = case_list

    def get_case_list(self):
        return self.__case_list

    def get_animal_name(self):
        """
        Returns the name of the animal this case belongs to
        """
        return self.__animal_name

    def import_case(self, data):
        """
        Creates the case from that native experimental data

        Arguments:
              data - a dictionary containing three keys:
                'filename' - full name of the train head
                'filetype' - 'compressed' or 'stream' depending on the data type
        """
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

    def load_case(self, xml, pathname=None):
        """
        Loads cases from the hard disk

        Arguments:
            xml - XML elements that corresponds to the case
            pathname - folder where case files are located
        """
        if pathname is not None:
            self.set_properties(pathname=pathname)
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
        if xml.find('roi-list') is not None:
            self.__load_roi(xml.find('roi-list'))
        else:
            self.__roi_list = RoiList()
        if xml.find('traces') is not None:
            self.__load_traces(xml.find('traces'))
        else:
            self.__traces_list = []
        if xml.find("results"):
            self.__load_results(xml.find("results"))
        else:
            self.__result_list = []

    def __read_file_list(self, xml, element_name):
        filelist = []
        for file_element in xml.findall(element_name):
            filelist.append(file_element.text)
        return filelist

    def __load_roi(self, xml):
        self.__roi_list.load(xml)

    def __load_traces(self, traces_list):
        self.__traces_list = []
        for trace_info in traces_list.findall("trace"):
            trace = Traces()
            trace.set_animal_name(trace_info.attrib['animal_name'])
            trace.set_case_name(self['short_name'])
            trace.set_roi_name(trace_info.attrib['roi_name'])
            trace.set_prefix_name(trace_info.attrib['prefix_name'])
            trace.set_postfix_name(trace_info.attrib['postfix_name'])
            trace.set_train_properties(trace_info.find('train-properties').attrib)
            trace.set_synchronization_properties(trace_info.find('synchronization-properties').attrib)
            trace.set_isoline_properties(trace_info.find('isoline-properties').attrib)
            output_file = os.path.join(self['pathname'], trace_info.attrib['src'])
            trace.set_output_file(output_file)
            self.__traces_list.append(trace)

    def __load_results(self, parent_xml):
        self.__result_list = []
        for result_element in parent_xml.findall("result"):
            result_name = result_element.attrib["name"]
            result_type = result_element.attrib["type"]
            result_class = imaging_data_classes[result_type]
            json_filename = os.path.join(self["pathname"], result_name + ".json")
            try:
                result = result_class(json_filename)
                self.__result_list.append(result)
            except FileNotFoundError:
                pass

    def save_case(self, parent_xml):
        """
        adds the case to the XML element

        Arguments:
            parent_xml - an elements where the case info shall be added to. The function will add the <case> element
            inside the parent_xml
        """
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
        roi_xml = self.__roi_list.save()
        roi_xml.tail = "\n"
        root.append(roi_xml)
        self.__save_trace_list(root)
        if len(self.__result_list) > 0:
            self.__save_result_list(root)

    def __save_trace_list(self, root):
        if self.traces_exist():
            traces = ET.SubElement(root, "traces")
            traces.text = "\n"
            traces.tail = "\n"
            for trace in self['traces']:
                trace_element = ET.SubElement(traces, "trace",
                                              animal_name=trace.get_animal_name(),
                                              fullname=trace.get_fullname(),
                                              prefix_name=trace.get_prefix_name(),
                                              postfix_name=trace.get_postfix_name(),
                                              roi_name=trace.get_roi_name(),
                                              src=os.path.split(trace.get_output_file())[1])
                trace_element.text = "\n"
                trace_element.tail = "\n"
                properties = trace.get_train_properties()
                for key, value in properties.items():
                    properties[key] = str(value)
                ET.SubElement(trace_element, "train-properties", properties).tail = "\n"
                properties = trace.get_synchronization_properties()
                for key, value in properties.items():
                    properties[key] = str(value)
                ET.SubElement(trace_element, "synchronization-properties", properties).tail = "\n"
                properties = trace.get_isoline_properties()
                for key, value in properties.items():
                    properties[key] = str(value)
                ET.SubElement(trace_element, "isoline-properties", properties).tail = "\n"

    def __save_result_list(self, parent_xml):
        result_list_xml = ET.SubElement(parent_xml, "results")
        result_list_xml.text = "\n"
        result_list_xml.tail = "\n"
        for data in self.data():
            type = data.__class__.__name__
            name = data.get_full_name()
            filename = data.get_filename()
            result_xml = ET.SubElement(result_list_xml, "result", {
                "type": type,
                "name": name,
                "src": filename
            })
            result_xml.tail = "\n"

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
        """
        Sets the main properties to the case

        Arguments:
            the keyword arguments like property_name=property_value
        The detailed information on property names is given in help pn __setitem__ function
        """
        for property_name in self.property_names:
            if property_name in kwargs:
                self.__properties[property_name] = copy.deepcopy(kwargs[property_name])

    def __getitem__(self, key):
        """
        Returns the property with a given name

        key is a string with one of the following meanings:
            "roi" - ROI list
            "traces" - all traces revealed without autoreading
            any other text properties are given in property_names global variable
        """
        if key == "roi":
            return self.__roi_list
        if key == "traces":
            return self.__traces_list
        if key in self.property_names:
            return self.__properties[key]
        else:
            raise IndexError("Subscript index doesn't refer to the valid property name")

    def __setitem__(self, key, value):
        """
        Sets a certain property.

        key is a string with one of the following meanings:
            "roi" - ROI list
            "traces" - all traces revealed without autoreading
            any other text properties are given in property_names global variable
        """
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
        s += "Traces: {0} traces".format(len(self.__traces_list))
        return s

    def get_traces_list(self):
        """
        Returns list of all traces revealed without autoprocess option
        """
        return self.__traces_list

    def get_traces_number(self):
        """
        Returns total number of traces revealed without using autoprocess
        """
        return len(self.__traces_list)

    def get_discarded_list(self):
        """
        Returns the list of all native data files associated with a particular case
        """
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

    def native_data_files_exist(self):
        """
        Returns True if native data exist and valid
        """
        if self['native_data_files'] is None:
            return False
        fullfile = os.path.join(self['pathname'], self['native_data_files'][0])
        try:
            train = sfiles.StreamFileTrain(fullfile, "traverse")
            train.open()
            train.close()
            return True
        except sfiles.IoError:
            return False

    def compressed_data_files_exist(self):
        """
        Returns True if compressed data files exist and valid
        """
        if self['compressed_data_files'] is None:
            return False
        fullfile = os.path.join(self['pathname'], self['compressed_data_files'][0])
        train = sfiles.CompressedFileTrain(fullfile, "traverse")
        try:
            train.open()
            train.close()
            return True
        except sfiles.IoError:
            return False

    def roi_exist(self):
        """
        Returns True if at least one ROI is given
        """
        return len(self.__roi_list) > 0

    def traces_exist(self):
        """
        Returns True if traces obtained manually exist
        """
        return self.get_traces_number() > 0

    def add_data(self, data):
        """
        Adds results to the case

        Arguments:
             data - the object that represents results. The object shall be an instance of the ImagingData
        """
        self.__result_list.append(data)

    def delete_data(self, data_name):
        """
        Deletes data from the list. Also, deletes JSON and NPZ files

        Arguments:
            data_name - name of the data to delete
        """
        current_data = None
        for data in self.data():
            if data.get_full_name() == data_name:
                current_data = data
        if current_data is None:
            raise IndexError("The argument doesn't refer to a valid data name")
        base_file = os.path.join(self["pathname"], data_name)
        json_file = base_file + ".json"
        npz_file = base_file + ".npz"
        try:
            os.unlink(json_file)
        except Exception as err:
            print("Failed to delete {0} due to {1}".format(json_file, err))
        try:
            os.unlink(npz_file)
        except Exception as err:
            print("Failed to delete {0} due to {1}".format(npz_file, err))
        self.__result_list.remove(current_data)

    def get_data(self, data_name):
        """
        Returns the data object that corresponds to a certain data name
        """
        for data in self.data():
            if data.get_full_name() == data_name:
                return data

    def data_exists(self):
        """
        Returns True if at least one processing result is added or loaded
        """
        return len(self.__result_list) > 0

    def data(self):
        """
        Returns iterator over all data participated in the analysis
        """
        return iter(self.__result_list)

    def auto_traces_exist(self):
        """
        Returns True if at least one trace revealed after 'Autoprocess' option exist, False otherwise
        """
        exist = False
        for data in self.data():
            if data.__class__ == ImagingSignal:
                exist = True
                break
        return exist

    def averaged_maps_exist(self):
        """
        Returns True if at least one averaged map exist for this case
        """
        exist = False
        for data in self.data():
            if data.__class__ == ImagingMap:
                exist = True
                break
        return exist

    def delete_native_files(self):
        folder_name = self['pathname']
        if self['native_data_files'] is not None:
            for native_file in self['native_data_files']:
                native_full_name = os.path.join(folder_name, native_file)
                os.unlink(native_full_name)
        self['native_data_files'] = None

    def delete_compressed_files(self):
        folder_name = self['pathname']
        if self['compressed_data_files'] is not None:
            for compressed_file in self['compressed_data_files']:
                compressed_full_name = os.path.join(folder_name, compressed_file)
                os.unlink(compressed_full_name)
        self['compressed_data_files'] = None
