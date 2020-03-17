# -*- coding: utf-8

import numpy as np
from ihna.kozhukhov.imageanalysis import ImagingMap
from .datatodataprocessor import DataToDataProcessor


class SpatialFilterDlg(DataToDataProcessor):

    def _get_processor_title(self):
        return "Spatial filter"

    def _check_input_data(self):
        if not isinstance(self._input_data, ImagingMap):
            raise ValueError("The input shall be complex imaging map")
        if self._input_data.get_data().dtype != np.complex:
            raise ValueError("The input map shall be complex imaging map")
