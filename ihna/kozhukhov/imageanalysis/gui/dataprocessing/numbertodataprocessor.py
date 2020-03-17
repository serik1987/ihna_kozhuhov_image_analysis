# -*- coding: utf-8

import wx
from .DataProcessDlg import DataProcessDlg


class NumberToDataProcessor(DataProcessDlg):

    def _get_processor_title(self):
        return "Sample number-to-data processor"

    def _check_input_data(self):
        pass
