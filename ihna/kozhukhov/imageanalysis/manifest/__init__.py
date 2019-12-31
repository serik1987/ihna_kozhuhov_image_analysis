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
