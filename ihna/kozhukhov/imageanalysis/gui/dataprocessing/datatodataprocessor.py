# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis import ImagingData
from .DataProcessDlg import DataProcessDlg


class DataToDataProcessor(DataProcessDlg):

    def _get_processor_title(self):
        return "Sample data-to-data processor"

    def _check_input_data(self):
        if not isinstance(self._input_data, ImagingData):
            raise ValueError("Please, select an appropriate imaging data for processing")
