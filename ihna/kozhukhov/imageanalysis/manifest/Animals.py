# -*- coding: utf-8

import os
import os.path
import shutil
from .animal import Animal
import xml.etree.ElementTree as ET

class Animals:
    """
    stores general information about all animals

        Operations:
            Animals(working_dir) - creates an empty animal list if the animal list has not been created
                                   OR: loads the animal list from the manifest.xml if the manifest has already been
                                   created
            the object is iterable. Iterations over the Animals object will given instances of the Animal objects
            the object is subscriptable:
            animals[specimen] will access to the particular specimen
            animals[specimen] = Animal(folder_name) - use this one in order to add the animal with a particular folder
            del animals[specimen] may be used to delete the specimen with all files belonging to this specimen
            print(animals) prints information about all animals as a table

        Attributes:
            manifest_fullname - full path to the iman-manifest.xml

        Other functions:
            animals.save() - to save the animal list to the hard disk drive
    """

    __working_dir = None
    __animal_list = None
    __animals_manifest = "iman-manifest.xml"
    __current_iterator = None

    def __init__(self, working_dir):
        self.__working_dir = working_dir
        self.__animal_list = dict()
        if os.path.isfile(self.manifest_fullname):
            self.load()

    def __getattr__(self, item):
        if item == "manifest_fullname":
            return os.path.join(self.__working_dir, self.__animals_manifest)
        else:
            raise AttributeError(item)

    def load(self):
        tree = ET.parse(self.manifest_fullname)
        root = tree.getroot()
        for animal_element in root.findall("animal"):
            animal = Animal(animal_element)
            specimen = animal['specimen']
            fullname = os.path.join(self.__working_dir, animal['folder_name'])
            animal['folder_full_name'] = fullname
            self[specimen] = animal

    def save(self):
        root = ET.Element("animals")
        root.text = "\n"
        for animal in self:
            ET.SubElement(root, "animal", attrib={
                "folder_name": animal['folder_name'],
                "specimen": animal['specimen'],
                "conditions": animal['conditions'],
                "recording_site": animal['recording_site']
            }).tail = "\n"
        tree = ET.ElementTree(root)
        tree.write(self.manifest_fullname, encoding="utf-8", xml_declaration=True)

    def __getitem__(self, key):
        return self.__animal_list[key]

    def __setitem__(self, key, value):
        if isinstance(key, str) and isinstance(value, Animal):
            self.__animal_list[key] = value
            value['folder_full_name'] = os.path.join(self.__working_dir, value['folder_name'])
            if value.is_new_animal():
                os.mkdir(value['folder_full_name'])
        else:
            raise AttributeError("the key/value pair is wrong")

    def __delitem__(self, key):
        deleting_animal = self[key]
        del self.__animal_list[key]
        full_dir_name = deleting_animal['folder_full_name']
        shutil.rmtree(full_dir_name)

    def __str__(self):
        s = "Specimen\tFolder name\tConditions\tRecording site\n"
        for animal in self:
            s += "{0:8}\t{1:11}\t{2:10}\t{3:14}\n".format(animal['specimen'], animal['folder_name'],
                                                          animal['conditions'], animal['recording_site'])
        return s

    def __iter__(self):
        self.__current_iterator = iter(self.__animal_list)
        return self

    def __next__(self):
        key = self.__current_iterator.__next__()
        return self.__animal_list[key]

    def replace_key(self, old_key, new_key):
        if old_key != new_key:
            self.__animal_list[new_key] = self.__animal_list[old_key]
            del self.__animal_list[old_key]
