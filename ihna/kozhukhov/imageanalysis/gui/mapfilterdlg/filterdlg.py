# -*- coding: utf-8

import numpy as np
from scipy.signal import freqz
import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from ihna.kozhukhov.imageanalysis.gui.mapfilterdlg import filters


class FilterDlg(wx.Dialog):

    __sample_rate = None

    __filter_type_box = None
    __panel = None
    __coefficients = None
    __filter_editors = None

    __fig1 = None
    __ax1 = None
    __fig2 = None
    __ax2 = None
    __canvas = None

    __order_info = None
    __delay_info = None
    __transition_band_info = None

    def __init__(self, parent, sample_rate,
                 default_transition_band=0.05,
                 default_bandpass_loss=0.04,
                 default_attenuation=50,
                 default_bandpass_ripples=0.04,
                 default_min_attenuation=50,
                 default_order=4):
        super().__init__(parent, title="Filter properties", size=(800, 600))
        self.__sample_rate = sample_rate
        panel = wx.Panel(self)
        self.__panel = panel
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.HORIZONTAL)
        left_layout = wx.BoxSizer(wx.VERTICAL)

        filter_type_layout = wx.BoxSizer(wx.HORIZONTAL)
        filter_type_caption = wx.StaticText(panel, label="Filter type")
        filter_type_layout.Add(filter_type_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        self.__filter_type_box = wx.Choice(panel, choices=list(filters.keys()), style=wx.CB_SORT)
        self.__filter_type_box.Bind(wx.EVT_CHOICE, lambda event: self.select_current())
        filter_type_layout.Add(self.__filter_type_box, 1, wx.EXPAND)

        left_layout.Add(filter_type_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__filter_editors = {}
        default_filter = None
        for filter_name, FilterEditor in filters.items():
            if FilterEditor is None:
                print("TO-DO: Implement editor for", filter_name)
            else:
                editor = FilterEditor(self, panel, sample_rate, default_transition_band, default_bandpass_loss,
                                      default_attenuation,
                                      default_bandpass_ripples, default_min_attenuation, default_order)
                self.__filter_editors[filter_name] = editor
                editor.hide()
                left_layout.Add(editor, 0, wx.EXPAND)
            if default_filter is None:
                default_filter = filter_name

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        apply = wx.Button(panel, label="Apply")
        apply.Bind(wx.EVT_BUTTON, lambda event: self.apply())
        buttons.Add(apply, 0, wx.RIGHT, 5)
        ok = wx.Button(panel, label="Continue")
        ok.Bind(wx.EVT_BUTTON, lambda event: self.set_filter())
        buttons.Add(ok, 0, wx.RIGHT, 5)
        cancel = wx.Button(panel, label="Cancel")
        cancel.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_CANCEL))
        buttons.Add(cancel, 0)
        left_layout.Add(buttons, 0, wx.ALIGN_CENTER | wx.BOTTOM, 5)

        self.__order_info = wx.StaticText(panel)
        left_layout.Add(self.__order_info, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__delay_info = wx.StaticText(panel)
        left_layout.Add(self.__delay_info, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__transition_band_info = wx.StaticText(panel)
        left_layout.Add(self.__transition_band_info, 0, wx.EXPAND)

        main_layout.Add(left_layout, 0, wx.EXPAND | wx.RIGHT, 10)
        right_layout = wx.Panel(panel)

        self.__fig1 = Figure()
        self.__ax1 = self.__fig1.add_subplot(211)
        self.__ax1.set_ylabel("Magnitude, dB")
        self.__ax2 = self.__fig1.add_subplot(212)
        self.__ax2.set_ylabel("Phase shift, rad")
        self.__ax2.set_xlabel("Frequency, Hz")

        canvas = FigureCanvas(right_layout, -1, self.__fig1)
        canvas_sizer = wx.BoxSizer(wx.VERTICAL)
        canvas_sizer.Add(canvas, 1, wx.GROW)
        right_layout.SetSizer(canvas_sizer)

        main_layout.Add(right_layout, 1, wx.EXPAND)
        general_layout.Add(main_layout, 1, wx.EXPAND | wx.ALL, 10)
        panel.SetSizer(general_layout)
        self.__canvas = canvas
        self.select(default_filter)

    def select(self, selected_name):
        for filter_name, filter_editor in self.__filter_editors.items():
            if filter_name == selected_name:
                filter_editor.show()
            else:
                filter_editor.hide()
        n = self.__filter_type_box.FindString(selected_name)
        self.__filter_type_box.SetSelection(n)
        self.__panel.Layout()
        self.Layout()
        if self.__fig1 is not None:
            self.__fig1.tight_layout()
        if self.__canvas is not None:
            self.__canvas.draw()

    def select_current(self):
        filter_name = self.__filter_type_box.GetStringSelection()
        self.select(filter_name)

    def apply(self):
        from ihna.kozhukhov.imageanalysis.gui import MainWindow
        try:
            filter_name = self.__filter_type_box.GetStringSelection()
            filter_editor = self.__filter_editors[filter_name]
            b, a = filter_editor.get_coefficients()
            W, h = freqz(b, a, fs=self.__sample_rate)
            amplitudes = 20 * np.log10(np.abs(h))
            phases = np.unwrap(np.angle(h))
            self.__ax1.clear()
            self.__ax1.plot(W, amplitudes)
            self.__ax1.set_xscale("log")
            self.__ax1.set_ylabel("Gain, dB")
            self.__ax1.set_ylim((-100, 0))
            self.__ax1.grid(True, which="both")
            self.__ax2.clear()
            self.__ax2.plot(W, phases)
            self.__ax2.set_xscale("log")
            self.__ax2.set_ylabel("phase, rad")
            self.__ax2.set_xlabel("Frequency, Hz")
            self.__ax2.grid(True)
            self.__canvas.draw()
            # self.__order_info.SetLabel("Filter order: %d" % filter_editor.get_current_order())
        except Exception as err:
            MainWindow.show_error_message(self, err, "Apply filter parameters")

    def set_filter(self):
        from ihna.kozhukhov.imageanalysis.gui import MainWindow
        try:
            filter_name = self.__filter_type_box.GetStringSelection()
            filter_editor = self.__filter_editors[filter_name]
            self.__coefficients = filter_editor.get_coefficients()
            self.EndModal(wx.ID_OK)
        except Exception as err:
            MainWindow.show_error_message(self, err, "Set the filter")

    def get_filter_coefficients(self):
        if self.__coefficients is None:
            raise AttributeError("Please, set filter properties by pressing 'Open filter properties' button")
        else:
            return self.__coefficients
