# -*- coding: utf8

import os
import wx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
from ihna.kozhukhov.imageanalysis import ImagingMap


class MapViewerDlg(wx.Dialog):

    __figure = None
    __full_name = None

    def __init__(self, parent, complexMap: ImagingMap):
        super().__init__(parent, title="Map view: " + complexMap.get_full_name(), size=(800, 500))
        panel = wx.Panel(self)
        general_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        figure_panel = wx.Panel(panel, size=(800, 300))
        figure = Figure()
        self._plot_graphs(figure, complexMap)
        canvas = FigureCanvas(panel, -1, figure)
        figure_sizer = wx.BoxSizer(wx.VERTICAL)
        figure_sizer.Add(canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        figure_panel.SetSizer(figure_sizer)
        figure.tight_layout()
        main_sizer.Add(figure_panel, 1, wx.BOTTOM | wx.EXPAND, 10)

        features = "; ".join(["%s: %s" % (key, value) for key, value in complexMap.get_features().items()])
        features_box = wx.StaticText(panel, label=features)
        features_box.SetSizeHints((800, 300))
        main_sizer.Add(features_box, 0, wx.BOTTOM | wx.EXPAND, 10)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(panel, label="Save and close")
        button.Bind(wx.EVT_BUTTON, lambda event: self.Close())
        button_sizer.Add(button, 0)
        main_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER)

        general_sizer.Add(main_sizer, 1, wx.EXPAND | wx.ALL, 10)
        panel.SetSizerAndFit(general_sizer)
        self.Centre()
        self.Fit()

        self.__figure = figure
        self.__full_name = complexMap.get_full_name()

    def _plot_graphs(self, figure, data):
        raise NotImplementedError("_plot_graphs")

    def save_png(self, folder_name):
        filename = os.path.join(folder_name, self.__full_name + ".png")
        self.__figure.savefig(filename)
