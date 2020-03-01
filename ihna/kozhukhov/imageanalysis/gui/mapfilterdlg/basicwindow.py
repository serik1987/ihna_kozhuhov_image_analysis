# -*- coding: utf-8

import numpy as np
from scipy.stats import linregress
import wx
from ihna.kozhukhov.imageanalysis.tracereading import TraceReader
from ihna.kozhukhov.imageanalysis.accumulators import MapFilter
from ihna.kozhukhov.imageanalysis.gui.readingprogressdlg import ReadingProgressDialog
from ihna.kozhukhov.imageanalysis.gui.frameaccumulatordlg import FrameAccumulatorDlg
from .filterdlg import FilterDlg


class BasicWindow(FrameAccumulatorDlg):

    __filter_type_box = None
    __filter_editors = None
    __parent = None

    __fs = None
    __filter_dlg = None
    __filter_dlg_applied = None

    def __init__(self, parent, train):
        super().__init__(parent, train, "Map filter")

    def _get_accumulator_class(self):
        return MapFilter

    def _create_frame_accumulator_box(self, parent):
        self.__parent = parent
        self.__calc_fs()
        filter_box = wx.StaticBoxSizer(wx.VERTICAL, parent, label="Filter properties")
        filter_layout = wx.BoxSizer(wx.VERTICAL)

        filter_type_button = wx.Button(parent, label="Open filter properties")
        filter_type_button.Bind(wx.EVT_BUTTON, lambda event: self.__open_filter_dlg())
        filter_layout.Add(filter_type_button)

        self.__filter_dlg = FilterDlg(self.__parent, self.__fs)
        self.__filter_dlg_applied = False

        filter_box.Add(filter_layout, 1, wx.ALL | wx.EXPAND, 5)
        return filter_box

    def __calc_fs(self):
        reader = TraceReader(self.get_train())
        reader.add_pixel(('TIME', 0))
        pbar = ReadingProgressDialog(self.__parent, "Calculation of sample rate", 1000,
                                     "Calculation of sample rate")
        reader.progress_bar = pbar
        try:
            reader.read()
        except Exception as err:
            print("PY Time vector reading failed")
            pbar.done()
            raise err
        pbar.done()
        times = reader.get_trace(0) * 1e-3
        timestamps = np.arange(times.size)
        result = linregress(timestamps, times)
        self.__fs = 1.0 / result[0]

    def __open_filter_dlg(self):
        if self.__filter_dlg.ShowModal() == wx.ID_OK:
            self.__filter_dlg_applied = True

    def get_sample_rate(self):
        return self.__fs

    def create_accumulator(self):
        map_filter = super().create_accumulator()
        b, a = self.__filter_dlg.get_filter_coefficients()
        map_filter.set_filter(b, a)
        return map_filter
