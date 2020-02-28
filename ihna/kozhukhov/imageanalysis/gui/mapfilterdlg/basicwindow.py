# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.accumulators import MapFilter
from ihna.kozhukhov.imageanalysis.gui.frameaccumulatordlg import FrameAccumulatorDlg
from ihna.kozhukhov.imageanalysis.gui.mapfilterdlg import filters


class BasicWindow(FrameAccumulatorDlg):

    __filter_type_box = None
    __filter_editors = None
    __parent = None

    def __init__(self, parent, train):
        super().__init__(parent, train, "Map filter")

    def _get_accumulator_class(self):
        return MapFilter

    def _create_frame_accumulator_box(self, parent):
        self.__parent = parent
        filter_box = wx.StaticBoxSizer(wx.VERTICAL, parent, label="Filter properties")
        filter_layout = wx.BoxSizer(wx.VERTICAL)
        filter_type_layout = wx.BoxSizer(wx.HORIZONTAL)

        filter_type_caption = wx.StaticText(parent, label="Filter type")
        filter_type_layout.Add(filter_type_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        self.__filter_type_box = wx.Choice(parent, choices=list(filters.keys()), style=wx.CB_SORT)
        self.__filter_type_box.Bind(wx.EVT_CHOICE, lambda event: self.select_current())
        filter_type_layout.Add(self.__filter_type_box, 1, wx.EXPAND)

        filter_layout.Add(filter_type_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__filter_editors = {}
        default_filter = None
        for filter_name, FilterEditor in filters.items():
            if FilterEditor is None:
                print("TO-DO: Implement editor for", filter_name)
            else:
                editor = FilterEditor(self, parent)
                self.__filter_editors[filter_name] = editor
                editor.hide()
                filter_layout.Add(editor, 0, wx.EXPAND)
            if default_filter is None:
                default_filter = filter_name
        self.select(default_filter)

        filter_box.Add(filter_layout, 1, wx.ALL | wx.EXPAND, 5)
        return filter_box

    def select(self, selected_name):
        for filter_name, filter_editor in self.__filter_editors.items():
            if filter_name == selected_name:
                print(filter_name)
                filter_editor.show()
            else:
                filter_editor.hide()
        n = self.__filter_type_box.FindString(selected_name)
        self.__filter_type_box.SetSelection(n)
        self.Layout()
        self.__parent.Layout()

    def select_current(self):
        filter_name = self.__filter_type_box.GetStringSelection()
        self.select(filter_name)
