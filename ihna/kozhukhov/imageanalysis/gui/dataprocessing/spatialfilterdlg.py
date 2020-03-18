# -*- coding: utf-8

import numpy as np
import wx
from ihna.kozhukhov.imageanalysis import ImagingMap
from .datatodataprocessor import DataToDataProcessor


class SpatialFilterDlg(DataToDataProcessor):

    __radius_box = None
    __radius_big_box = None

    __radius_checkbox = None
    __radius_big_checkbox = None

    def _get_processor_title(self):
        return "Spatial filter"

    def _check_input_data(self):
        if not isinstance(self._input_data, ImagingMap):
            raise ValueError("The input shall be complex imaging map")
        if self._input_data.get_data().dtype != np.complex:
            raise ValueError("The input map shall be complex imaging map")

    def _get_default_minor_name(self):
        return "filt"

    def _place_additional_options(self, parent):
        additional_options = wx.BoxSizer(wx.VERTICAL)
        radius_layout = wx.BoxSizer(wx.HORIZONTAL)

        radius_caption = wx.StaticText(parent, label="Inner radius, px")
        radius_layout.Add(radius_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        self.__radius_box = wx.TextCtrl(parent)
        radius_layout.Add(self.__radius_box, 1, wx.EXPAND | wx.RIGHT, 5)

        self.__radius_checkbox = wx.CheckBox(parent, label="Off")
        self.__radius_checkbox.Bind(wx.EVT_CHECKBOX, lambda event: self.__switch_inner_radius())
        radius_layout.Add(self.__radius_checkbox, 0, wx.ALIGN_CENTER_VERTICAL)

        additional_options.Add(radius_layout, 0, wx.EXPAND | wx.BOTTOM, 5)
        radius_big_layout = wx.BoxSizer(wx.HORIZONTAL)

        radius_big_caption = wx.StaticText(parent, label="Outer radius, px")
        radius_big_layout.Add(radius_big_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        self.__radius_big_box = wx.TextCtrl(parent)
        radius_big_layout.Add(self.__radius_big_box, 1, wx.EXPAND | wx.RIGHT, 5)

        self.__radius_big_checkbox = wx.CheckBox(parent, label="Off")
        self.__radius_big_checkbox.Bind(wx.EVT_CHECKBOX, lambda event: self.__switch_outer_radius())
        radius_big_layout.Add(self.__radius_big_checkbox, 0, wx.ALIGN_CENTER_VERTICAL)

        additional_options.Add(radius_big_layout, 0, wx.EXPAND)
        return additional_options

    def __switch_inner_radius(self):
        if not self.__radius_checkbox.IsChecked():
            self.__radius_box.Enable(True)
        else:
            self.__radius_box.Enable(False)
            self.__radius_box.SetValue("")

    def __switch_outer_radius(self):
        if not self.__radius_big_checkbox.IsChecked():
            self.__radius_big_box.Enable(True)
        else:
            self.__radius_big_box.Enable(False)
            self.__radius_big_box.SetValue("")
