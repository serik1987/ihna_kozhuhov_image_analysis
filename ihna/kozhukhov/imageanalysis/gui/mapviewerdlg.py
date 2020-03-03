# -*- coding: utf8

import wx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
from ihna.kozhukhov.imageanalysis import ImagingMap


class MapViewerDlg(wx.Dialog):

    def __init__(self, parent, complexMap: ImagingMap):
        super().__init__(parent, title="Map view: " + complexMap.get_full_name(), size=(800, 500))
        panel = wx.Panel(self)
        general_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        figure_panel = wx.Panel(panel, size=(800, 300))
        figure = Figure()
        amplitude_axes = figure.add_subplot(121)
        amplitude_map = amplitude_axes.imshow(np.abs(complexMap.get_data()), cmap="gray")
        figure.colorbar(amplitude_map, ax=amplitude_axes)
        amplitude_axes.set_title("Amplitude map")
        phase_axes = figure.add_subplot(122)
        H = complexMap.get_harmonic()
        phase_data = np.angle(complexMap.get_data()) / H
        phase_data[phase_data < 0] += 2 * np.pi / H
        phase_data = 180 * phase_data / np.pi
        phase_max = 360 / H
        phase_map = phase_axes.imshow(phase_data, cmap="hsv", vmin=0, vmax=phase_max)
        phase_axes.set_title("Phase map")
        figure.colorbar(phase_map, ax=phase_axes)
        canvas = FigureCanvas(panel, -1, figure)
        figure_sizer = wx.BoxSizer(wx.VERTICAL)
        figure_sizer.Add(canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        figure_panel.SetSizer(figure_sizer)
        figure.tight_layout()
        main_sizer.Add(figure_panel, 1, wx.BOTTOM | wx.EXPAND, 10)

        features = "; ".join(["%s: %s" % (key, value) for key, value in complexMap.get_features().items()])
        features_box = wx.StaticText(panel, label=features)
        features_box.SetSizeHints((600, 300))
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
