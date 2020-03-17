# -*- coding: utf-8

import wx
from .numbertodataprocessor import NumberToDataProcessor


class MapFillterDlg(NumberToDataProcessor):

    def _get_processor_title(self):
        return "New map with predefined values"

    def _get_default_minor_name(self):
        return "fill"
