# -*- coding: utf-8
"""
This package may be used to retrieve general information about the data by reading the iman-manifest.xml files

iman-manifest.xml contains general information about all animals.
Despite of this, manifest.xml within the each folder contains general information about all cases stores within
this folder
"""

from .Animals import Animals
from .animal import Animal
from .caseslist import CasesList
from .case import Case
from .roi import Roi
from .simpleroi import SimpleRoi
from .roilist import RoiList
from .complexroi import ComplexRoi
from .filter import Filter
from .animalfilter import AnimalFilter
from .casefilter import CaseFilter
