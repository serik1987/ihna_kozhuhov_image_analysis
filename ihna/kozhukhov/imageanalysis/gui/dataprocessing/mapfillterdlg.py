# -*- coding: utf-8

import wx
from .numbertodataprocessor import NumberToDataProcessor


class MapFillterDlg(NumberToDataProcessor):

    def _get_processor_title(self):
        return "New map with predefined values"
