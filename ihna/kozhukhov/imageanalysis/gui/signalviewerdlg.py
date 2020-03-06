# -*- coding: utf-8

import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from ihna.kozhukhov.imageanalysis import ImagingSignal


class SignalViewerDlg(wx.Dialog):

    __ask_names = None
    __prefix_name_box = None
    __postfix_name_box = None
    __signal = None

    def __init__(self, parent, signal: ImagingSignal, ask_names=False):
        super().__init__(parent, title="Signal view: " + signal.get_full_name(), size=(800, 1000))
        self.__signal = signal
        self.__ask_names = ask_names
        panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)

        figure_panel = wx.Panel(self, size=(800, 300))
        figure_panel.SetBackgroundColour("red")
        figure = Figure()
        signal_axes = figure.add_subplot(121)
        signal_axes.plot(signal.get_times(), signal.get_values())
        signal_axes.set_xlabel("Time, ms")
        signal_axes.set_ylabel("Signal, %")
        spectrum_axes = figure.add_subplot(122)
        spectrum_axes.plot(signal.get_frequencies(), signal.get_spectrum())
        spectrum_axes.set_xlabel("Frequency, Hz")
        spectrum_axes.set_ylabel("Power")
        spectrum_axes.set_xscale("log")
        figure_canvas = FigureCanvas(figure_panel, -1, figure)
        figure.tight_layout()
        figure_sizer = wx.BoxSizer(wx.VERTICAL)
        figure_sizer.Add(figure_canvas, 1, wx.GROW)
        figure_panel.SetSizer(figure_sizer)
        main_layout.Add(figure_panel, 1, wx.EXPAND | wx.BOTTOM, 10)

        features = "; ".join(["%s: %s" % (key, value) for key, value in signal.get_features().items()])
        if ask_names:
            height = 150
        else:
            height = 200
        features_box = wx.StaticText(panel, label=features)
        features_box.SetSizeHints((800,  height))
        main_layout.Add(features_box, 0, wx.BOTTOM | wx.EXPAND, 10)

        if ask_names:
            all_names_sizer = wx.BoxSizer(wx.HORIZONTAL)
            prefix_name_caption = wx.StaticText(panel, label="Prefix name")
            all_names_sizer.Add(prefix_name_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

            self.__prefix_name_box = wx.TextCtrl(panel)
            all_names_sizer.Add(self.__prefix_name_box, 1, wx.RIGHT | wx.EXPAND, 10)

            postfix_name_caption = wx.StaticText(panel, label="Postfix name")
            all_names_sizer.Add(postfix_name_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

            self.__postfix_name_box = wx.TextCtrl(panel)
            all_names_sizer.Add(self.__postfix_name_box, 1, wx.EXPAND)

            main_layout.Add(all_names_sizer, 0, wx.BOTTOM | wx.EXPAND, 5)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button = wx.Button(panel, label="Save and close")
        button.Bind(wx.EVT_BUTTON, lambda event: self.finalize_dlg())
        button_sizer.Add(button, 0)
        main_layout.Add(button_sizer, 0, wx.ALIGN_CENTER)

        general_layout.Add(main_layout, 1, wx.EXPAND | wx.ALL, 10)
        panel.SetSizerAndFit(general_layout)
        self.Centre()
        self.Fit()

    def finalize_dlg(self):
        if self.__ask_names:
            prefix = self.__prefix_name_box.GetValue()
            postfix = self.__postfix_name_box.GetValue()
            animal_name, short_name = self.__signal.get_features()["major_name"].split("_")
            self.__signal.get_features()["major_name"] = "%s_%s%s%s" % (animal_name, prefix, short_name, postfix)
        self.Close()
