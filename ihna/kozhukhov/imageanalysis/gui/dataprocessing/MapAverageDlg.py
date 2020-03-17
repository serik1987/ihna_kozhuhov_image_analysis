# -*-  coding: utf-8

import wx
import numpy as np
from ihna.kozhukhov.imageanalysis import ImagingMap
from .datatonumberprocessor import DataToNumberProcessor


class MapAverageDlg(DataToNumberProcessor):

    def _get_processor_title(self):
        return "Map average value"

    def _check_input_data(self):
        if not isinstance(self._input_data, ImagingMap):
            raise ValueError("The average data can be computed for maps only")
        if self._input_data.get_data().dtype == np.complex:
            dlg = wx.MessageDialog(self._parent,
                                   "The average value for complex maps is some meaningless complex value",
                                   "Map average",
                                   wx.OK | wx.CENTRE | wx.ICON_WARNING)
            dlg.ShowModal()
