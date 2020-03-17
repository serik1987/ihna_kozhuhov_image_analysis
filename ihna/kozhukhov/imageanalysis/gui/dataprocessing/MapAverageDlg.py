# -*-  coding: utf-8

import wx
import numpy as np
from ihna.kozhukhov.imageanalysis import ImagingMap
from .datatonumberprocessor import DataToNumberProcessor


class MapAverageDlg(DataToNumberProcessor):

    def _get_processor_title(self):
        return "Map average value"

    def _check_input_data(self):
        return self._check_imaging_map(True)

    def _get_default_minor_name(self):
        return "average"
