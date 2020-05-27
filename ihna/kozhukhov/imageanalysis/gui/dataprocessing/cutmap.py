# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis import ImagingMap
from ihna.kozhukhov.imageanalysis.gui.complexmapviewerdlg import ComplexMapViewerDlg
from .datatodataprocessor import DataToDataProcessor


class CutMap(DataToDataProcessor):

    __roi_selector = None

    def _get_default_minor_name(self):
        return "cut"

    def _check_input_data(self):
        if not isinstance(self._input_data, ImagingMap):
            raise ValueError("The processor is available for imaging maps only")
        if len(self._considering_case['roi']) == 0:
            raise AttributeError("Please, specify at least one ROI using Use Case -> ROI manager facility")

    def _place_additional_options(self, parent):
        roi_names = [roi.get_name() for roi in self._considering_case['roi']]
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        caption = wx.StaticText(parent, label="ROI")
        sizer.Add(caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.__roi_selector = wx.Choice(parent, choices=roi_names, style=wx.CB_SORT)
        self.__roi_selector.SetSelection(0)
        sizer.Add(self.__roi_selector, 1, wx.EXPAND)
        return sizer

    def _process(self):
        roi_name = self.__roi_selector.GetStringSelection()
        roi = self._considering_case['roi'][roi_name]
        features = self._input_data.get_features().copy()
        features['minor_name'] = self.get_output_file()
        features['original_map'] = self._input_data.get_full_name()
        features['is_main'] = 'no'
        data = roi.apply(self._input_data.get_data())
        self._output_data = ImagingMap(features, data)

    def _get_result_viewer(self):
        return ComplexMapViewerDlg
