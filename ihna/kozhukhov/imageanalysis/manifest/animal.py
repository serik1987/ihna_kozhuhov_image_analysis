# -*- coding: utf-8

import xml.etree.ElementTree as ET


class Animal:
    """
    The class represents a single animal.

    Initialization:
        using methods of the Animals object is the preferrable way to create the class.
        Another ways are:
            Animal(folder_name) - creates the folder name corresponding to an animal and then creates an empty animal
            with no cases. The newly created Animal shall be inserted into Animals array
            Animal(xml) - will load the animal from the XML element passed as an argument

        Operations on animal:
            print(animal) - prints the general information about the animal, doesn't print the case list
            animal[field_name] = value_name - changing a certain field of the animal general information
            animal[field_name] - reading a certain field of the animal general information

        Field names:
            'folder_name' - name of the folder corresponding the animal
            'folder_full_name' - full name of the folder corresponding to the animal
            'specimen' - name of the specimen
            'cnoditions' - experimental conditions if otherwise is not defined by the case
            'recording_site' - the brain area recorded by the method
    """

    __animal_properties = None
    __new_animal = False

    def __init__(self, parameter):
        if isinstance(parameter, str):
            self.new(parameter)
        elif isinstance(parameter, ET.Element):
            self.load(parameter)
        else:
            raise AttributeError("Incorrect argument for the Animal constructor")

    def new(self, folder_name):
        self.__new_animal = True
        self.__animal_properties = {
            "folder_name": folder_name,
            "folder_full_name": None,
            "specimen": folder_name,
            "conditions": "",
            "recording_site": ""
        }

    def load(self, element):
        self.__new_animal = False
        self.__animal_properties = {
            "folder_name": element.attrib['folder_name'],
            "specimen": element.attrib['specimen'],
            "conditions": element.attrib['conditions'],
            "recording_site": element.attrib['recording_site'],
            "folder_full_name": None
        }

    def __str__(self):
        s = "Animal name: {0}\n\
Conditions: {1}\n\
Recording site: {2}\n\
Folder name: {3}\n\
Folder full name: {4}\n".format(self.__animal_properties['specimen'], self.__animal_properties['conditions'],
                                self.__animal_properties['recording_site'], self.__animal_properties['folder_name'],
                                self.__animal_properties['folder_full_name'])
        return s

    def __getitem__(self, item):
        return self.__animal_properties[item]

    def __setitem__(self, key, value):
        if key in ["folder_name", "folder_full_name", "specimen", "conditions", "recording_site"]:
            self.__animal_properties[key] = value

    def is_new_animal(self):
        """
        Returns True if there is not animal folder corresponding to the animal, False otherwise
        """
        return self.__new_animal

    def set_new_animal(self):
        """
        Sets the NEW_ANIMAL flag to True. If the animal will be added to the animal list, this will result to creating
        of a certain folder
        """
        self.__new_animal = True

    def clear_new_animal(self):
        """
        Clears the NEW_ANIMAL flag. This suppresses attempt to create an animal folder when the animal is added into
        the animal list
        """
        self.__new_animal = False
