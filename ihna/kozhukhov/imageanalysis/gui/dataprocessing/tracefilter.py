# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis import ImagingSignal
from ihna.kozhukhov.imageanalysis.traceprocessing import filter_trace
from ihna.kozhukhov.imageanalysis.gui.signalviewerdlg import SignalViewerDlg
from ihna.kozhukhov.imageanalysis.gui.mapfilterdlg.filterdlg import FilterDlg
from .datatodataprocessor import DataToDataProcessor


class TraceFilterDlg(DataToDataProcessor):

    __filter_button = None
    __filter_dlg = None

    def __init__(self, parent, input_data, considering_case):
        super().__init__(parent, input_data, considering_case)
        self.__filter_button = None
        self.__filter_dlg = None

    def _check_input_data(self):
        super()._check_input_data()
        if not isinstance(self._input_data, ImagingSignal):
            raise RuntimeError("This processor is suitable for traces only")

    def _get_default_minor_name(self):
        roi_name = self._input_data.get_features()["ROI"]
        return "tracefilt(%s)" % roi_name

    def _place_additional_options(self, parent):
        self.__filter_button = wx.Button(parent, label="Open filter properties")
        self.__filter_button.Bind(wx.EVT_BUTTON, self.open_filter_properties)

        return self.__filter_button

    def open_filter_properties(self, event=None):
        if self.__filter_dlg is None:
            sample_rate = self._input_data.get_sample_rate()
            self.__filter_dlg = FilterDlg(self, sample_rate)
        self.__filter_dlg.ShowModal()

    def _process(self):
        if self.__filter_dlg is None:
            raise RuntimeError("Please, press 'Open filter properties' button to set appropriate filter properties")
        b, a = self.__filter_dlg.get_filter_coefficients()
        self._output_data = filter_trace(self._input_data, b, a)
        self._output_data.get_features()["filter"] = self.__filter_dlg.get_filter_description()

    def _get_result_viewer(self):
        return SignalViewerDlg

