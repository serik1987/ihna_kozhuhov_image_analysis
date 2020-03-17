# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis import ImagingData
from .DataProcessDlg import DataProcessDlg


class DataToNumberProcessor(DataProcessDlg):

    def _get_processor_title(self):
        return "Sample data-to-number processor"

    def _check_input_data(self):
        if not isinstance(self._input_data, ImagingData):
            raise ValueError("This processor requires a single imaging data as an input")
