# -*- coding: utf-8

import wx


class ResultListDlg(wx.Dialog):

    __case = None

    def __init__(self, parent, case):
        self.__case = case
        super().__init__(parent, title=self.get_title(), size=(700, 500))
        main_panel = wx.Panel(self)
        general_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        upper_sizer = wx.BoxSizer(wx.HORIZONTAL)

        left_panel = wx.Panel(main_panel, size=(100, 20))
        left_panel.SetBackgroundColour("green")
        upper_sizer.Add(left_panel, 1, wx.EXPAND | wx.RIGHT, 10)

        right_panel = wx.Panel(main_panel, size=(100, 20))
        right_panel.SetBackgroundColour("black")
        upper_sizer.Add(right_panel, 1, wx.EXPAND)

        main_sizer.Add(upper_sizer, 1, wx.EXPAND | wx.BOTTOM, 10)
        middle_sizer = wx.Panel(main_panel, size=(100, 20))
        middle_sizer.SetBackgroundColour("blue")

        main_sizer.Add(middle_sizer, 0, wx.EXPAND)
        general_sizer.Add(main_sizer, 1, wx.EXPAND | wx.ALL, 10)
        main_panel.SetSizer(general_sizer)
        self.Centre()

    def get_case(self):
        return self.__case

    def get_title(self):
        return "%s: %s_%s" % (self._get_base_title(), self.__case.get_animal_name(), self.__case['short_name'])

    def _get_base_title(self):
        raise NotImplementedError("_get_base_title")
