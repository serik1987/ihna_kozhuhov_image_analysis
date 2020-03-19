# -*- coding: utf-8

import os.path
import xml.etree.ElementTree as ET
from ihna.kozhukhov.imageanalysis.sourcefiles import get_file_info
from .case import Case
from .casefilter import CaseFilter


class CasesList:
    """
    This class allows to deal with case list, load it from the manifest file and save it to the manifest file,
    add imported cases to the case list or delete the cases, access to the individual cases

    How to create it:
        cases_list = CasesList(animal) where animal is an instance of ihna.kozhukhov.imageanalysis.sourcefiles.Animal

    Operations:
        cases_list[short_name] or cases_fist[filename] - access to a particular case
        for case in cases_list:
            do_something(case)
        The case list is iterable if you need to iterate over all records

        del cases_list[short_name] or del cases_list[filename] will delete the case and remove all files associated
        with it
    """

    __corresponding_animal = None
    __all_cases = None
    __discarded_list = None
    __filter = None
    __animal_name = None

    def __init__(self, animal):
        self.__animal_name = animal['specimen']
        self.__corresponding_animal = animal
        if os.path.isfile(self.get_manifest_file()):
            self.load()
        else:
            self.__all_cases = []
            self.__discarded_list = [self.get_manifest_file()]
            self.__filter = CaseFilter(self)
        source_files = os.listdir(self.__corresponding_animal['folder_full_name'])
        for idx in range(len(source_files)):
            source_files[idx] = os.path.join(self.__corresponding_animal['folder_full_name'], source_files[idx])
        source_files.sort()
        valid_files = get_file_info(source_files, self.__discarded_list)[0]
        for valid_file in valid_files:
            self.__all_cases.append(Case(valid_file, animal_name=self.__animal_name, case_list=self))

    def load(self):
        self.__discarded_list = [self.get_manifest_file()]
        self.__all_cases = []
        pathname = self.__corresponding_animal['folder_full_name']
        root = ET.parse(self.get_manifest_file()).getroot()
        for case_element in root.findall("case"):
            case = Case(case_element, pathname=pathname, animal_name=self.__animal_name, case_list=self)
            self.__discarded_list.extend(case.get_discarded_list())
            self.__all_cases.append(case)
        self.__filter = CaseFilter(self, root.find("case-filter"))

    def save(self):
        root = ET.Element("caselist")
        root.text = "\n"
        for case in self:
            case.save_case(root)
        self.__filter.save(root, "case-filter")
        tree = ET.ElementTree(root)
        tree.write(self.get_manifest_file(), encoding="utf-8", xml_declaration=True)

    def get_manifest_file(self):
        return os.path.join(self.__corresponding_animal['folder_full_name'], "iman-manifest.xml")

    def __iter__(self):
        return iter(self.__all_cases)

    def __getitem__(self, key):
        for case in self.__all_cases:
            if case['short_name'] == key or case['filename'] == key:
                return case
        raise IndexError("The case with a given short_name of filename is not found")

    def __delitem__(self, key):
        deleting_case = None
        for case in self.__all_cases:
            if case['short_name'] == key or case['filename'] == key:
                deleting_case = case
                self.__all_cases.remove(case)
        filelist = deleting_case.get_discarded_list()
        for file in filelist:
            os.remove(file)

    def get_case_filter(self):
        """
        Returns the case filter set by the user
        """
        return self.__filter

    def get_all_cases(self):
        """
        (For internal usage only)
        """
        return self.__all_cases

    def get_animal_name(self):
        """
        Returns the animal to which the specimen belongs
        """
        return self.__animal_name
