# -*- coding: utf-8

import wx
from .accumulatordlg import AccumulatorDlg


class FrameAccumulatorDlg(AccumulatorDlg):

    __preprocess_filter_box = None
    __preprocess_filter_caption = None
    __preprocess_filter_radius_box = None
    __divide_by_average_box = None

    def _create_accumulator_box(self, parent):
        result = wx.BoxSizer(wx.VERTICAL)

        accumulator_box = wx.StaticBoxSizer(wx.VERTICAL, parent, label="Frame accumulation")
        layout = wx.BoxSizer(wx.VERTICAL)

        self.__preprocess_filter_box = wx.CheckBox(parent, label="Use preprocess spatial LPF")
        self.Bind(wx.EVT_CHECKBOX, lambda event: self.__setpreprocess_filter_radius_visibility(),
                  self.__preprocess_filter_box)
        layout.Add(self.__preprocess_filter_box, 0, wx.BOTTOM, 5)

        preprocess_filter_radius_sizer = wx.BoxSizer(wx.HORIZONTAL)
        preprocess_filter_radius_caption = wx.StaticText(parent, label="Filter radius, px")
        preprocess_filter_radius_sizer.Add(preprocess_filter_radius_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.__preprocess_filter_radius_box = wx.TextCtrl(parent)
        preprocess_filter_radius_sizer.Add(self.__preprocess_filter_radius_box, 1)
        layout.Add(preprocess_filter_radius_sizer, 1, wx.BOTTOM | wx.EXPAND, 5)
        self.__preprocess_filter_caption = preprocess_filter_radius_caption
        self.__setpreprocess_filter_radius_visibility()

        self.__divide_by_average_box = wx.CheckBox(parent, label="Divide by average")
        layout.Add(self.__divide_by_average_box, 0, wx.EXPAND)

        accumulator_box.Add(layout, 1, wx.EXPAND | wx.ALL, 10)
        result.Add(accumulator_box, 0, wx.EXPAND | wx.BOTTOM, 5)

        frame_accumulator_box = self._create_frame_accumulator_box(parent)
        result.Add(frame_accumulator_box, 0, wx.EXPAND)

        return result

    def __setpreprocess_filter_radius_visibility(self):
        checked = self.__preprocess_filter_box.IsChecked()
        self.__preprocess_filter_caption.Enable(checked)
        self.__preprocess_filter_radius_box.Enable(checked)

    def _create_frame_accumulator_box(self, parent):
        frame_accumulator_box = wx.Panel(parent, size=(100, 200))
        frame_accumulator_box.SetBackgroundColour("red")
        return frame_accumulator_box

    def create_accumulator(self):
        accumulator = super().create_accumulator()
        if self.__preprocess_filter_box.IsChecked():
            accumulator.preprocess_filter = True
            try:
                filter_radius = int(self.__preprocess_filter_radius_box.GetValue())
            except ValueError:
                raise ValueError("Filter radius shall be integer expressed in pixels")
            accumulator.preprocess_filter_radius = filter_radius
        else:
            accumulator.preprocess_filter = False
        accumulator.divide_by_average = self.__divide_by_average_box.IsChecked()
        return accumulator
