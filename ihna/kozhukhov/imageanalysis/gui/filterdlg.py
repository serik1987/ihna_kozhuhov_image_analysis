# -*- coding: utf-8

import wx


class FilterDlg(wx.Dialog):

    _filter = None
    _widget_titles = None
    __widgets = None

    def __init__(self, parent, base_filter, title):
        super().__init__(parent, title=title)
        self._filter = base_filter
        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)
        widget_sizer = wx.FlexGridSizer(2, 2, 5)

        captions = self.get_property_captions()
        self.__widgets = {}
        for property_name in self._filter.get_filter_fields():
            property_caption = wx.StaticText(main_panel, label=captions[property_name])
            widget_sizer.Add(property_caption, 0, wx.ALIGN_CENTER_VERTICAL)
            property_box = wx.TextCtrl(main_panel, value=self._filter[property_name])
            property_box.SetSizeHints(200, property_box.GetSize().GetHeight())
            widget_sizer.Add(property_box, 1, wx.EXPAND)
            self.__widgets[property_name] = property_box

        main_layout.Add(widget_sizer, 1, wx.EXPAND | wx.BOTTOM, 10)
        bottom_layout = wx.BoxSizer(wx.HORIZONTAL)

        ok = wx.Button(main_panel, label="OK")
        self.Bind(wx.EVT_BUTTON, lambda event: self.end_modal(), ok)
        bottom_layout.Add(ok, 0, wx.RIGHT, 5)

        cancel = wx.Button(main_panel, label="Cancel")
        self.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_CANCEL), cancel)
        bottom_layout.Add(cancel)

        main_layout.Add(bottom_layout, 0, wx.ALIGN_CENTER)
        general_layout.Add(main_layout, 1, wx.EXPAND | wx.ALL, 10)
        main_panel.SetSizerAndFit(general_layout)
        self.Centre()
        self.Fit()

    def get_property_captions(self):
        raise NotImplementedError("FilterDlg is fully abstract. Use its derivatives")

    def end_modal(self):
        for property_name in self._filter.get_filter_fields():
            property_value = self.__widgets[property_name].GetValue()
            self._filter[property_name] = property_value
        self.EndModal(wx.ID_OK)
