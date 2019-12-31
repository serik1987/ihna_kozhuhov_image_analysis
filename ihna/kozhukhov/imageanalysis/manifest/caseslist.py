# -*- coding: utf-8

import os.path
from ihna.kozhukhov.imageanalysis.sourcefiles import get_file_info
from .case import Case


class CasesList:
    """
    This class allows to deal with case list, load it from the manifest file and save it to the manifest file,
    add imported cases to the case list or delete the cases, access to the individual cases

    How to create it:
        cases_list = CasesList(animal) where aninal is an instance of ihna.kozhukhov.imageanalysis.sourcefiles.Animal

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

    def __init__(self, animal):
        self.__corresponding_animal = animal
        if os.path.isfile(self.get_manifest_file()):
            self.load()
        else:
            self.__all_cases = []
            self.__discarded_list = []
        source_files = os.listdir(self.__corresponding_animal['folder_full_name'])
        for idx in range(len(source_files)):
            source_files[idx] = os.path.join(self.__corresponding_animal['folder_full_name'], source_files[idx])
        source_files.sort()
        valid_files = get_file_info(source_files, self.__discarded_list)[0]
        self.__all_cases = []
        for valid_file in valid_files:
            self.__all_cases.append(Case(valid_file))

    def load(self):
        print("Loading the cases list from the manifest file")

    def save(self):
        print("Saving the cases list to the manifest file")

    def get_manifest_file(self):
        return os.path.join(self.__corresponding_animal['folder_full_name'], "iman-manifest.xml")

    def __iter__(self):
        return iter(self.__all_cases)

    def __getitem__(self, key):
        for case in self.__all_cases:
            if case['short_name'] == key or case['filename'] == key:
                return case
        raise IndexError("The case with a given short_name of filename is not found")
