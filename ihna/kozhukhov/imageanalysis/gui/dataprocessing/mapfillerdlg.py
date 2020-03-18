# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.mapprocessing import fill_map
from ihna.kozhukhov.imageanalysis.gui.amplitudemapviewerdlg import AmplitudeMapViewerDlg
from .numbertodataprocessor import NumberToDataProcessor


class MapFillerDlg(NumberToDataProcessor):

    __map_size_x_box = None
    __map_size_y_box = None
    __filling_value = None

    def _get_processor_title(self):
        return "New map with predefined values"

    def _get_default_minor_name(self):
        return "fill"

    def _place_additional_options(self, parent):
        additional_options = wx.BoxSizer(wx.VERTICAL)

        map_size_x_layout = wx.BoxSizer(wx.HORIZONTAL)
        map_size_x_caption = wx.StaticText(parent, label="Map size, X, px")
        map_size_x_layout.Add(map_size_x_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.__map_size_x_box = wx.TextCtrl(parent, value="512")
        map_size_x_layout.Add(self.__map_size_x_box, 1, wx.EXPAND)
        additional_options.Add(map_size_x_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

        map_size_y_layout = wx.BoxSizer(wx.HORIZONTAL)
        map_size_y_caption = wx.StaticText(parent, label="Map size, Y, px")
        map_size_y_layout.Add(map_size_y_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.__map_size_y_box = wx.TextCtrl(parent, value="512")
        map_size_y_layout.Add(self.__map_size_y_box, 1, wx.EXPAND)
        additional_options.Add(map_size_y_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

        filling_value_layout = wx.BoxSizer(wx.HORIZONTAL)
        filling_value_caption = wx.StaticText(parent, label="Filling value")
        filling_value_layout.Add(filling_value_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.__filling_value = wx.TextCtrl(parent, value="0")
        filling_value_layout.Add(self.__filling_value, 1, wx.EXPAND)
        additional_options.Add(filling_value_layout, 0, wx.EXPAND)

        return additional_options

    def _process(self):
        major_name = self._get_desired_major_name()
        minor_name = self.get_output_file()

        try:
            map_size_x = int(self.__map_size_y_box.GetValue())
            if map_size_x <= 0:
                raise ValueError("Map size on X shall be positive")
        except ValueError:
            raise ValueError("Map size on X must be integer")

        try:
            map_size_y = int(self.__map_size_y_box.GetValue())
            if map_size_y <= 0:
                raise ValueError("Map size on Y shall be positive")
        except ValueError:
            raise ValueError("Map size on Y must be integer")

        try:
            fill_value = float(self.__filling_value.GetValue())
        except ValueError:
            raise ValueError("The filling value must be a number")

        self._output_data = fill_map(major_name, minor_name, map_size_x, map_size_y, fill_value)

    def _get_result_viewer(self):
        return AmplitudeMapViewerDlg
