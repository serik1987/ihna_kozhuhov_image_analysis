#!/usr/bin/env python3
# -*- coding: utf-8

import os
import xml.etree.ElementTree as ET
from ihna.kozhukhov.imageanalysis.manifest import *

if __name__ == "__main__":
    working_dir = "/home/serik1987/vasomotor-oscillations"
    roi_file = os.path.join(working_dir, "new_project/sample_roi.xml")
    roi_file2 = os.path.join(working_dir, "new_project/sample_complex_roi.xml")
    animals = Animals(working_dir)
    animal = animals['c022']
    cases = CasesList(animal)
    case = cases['00']

    roi_list = RoiList()
    tree = ET.parse(roi_file2)
    element = tree.getroot()
    roi_list.load(element)
    print(roi_list)
