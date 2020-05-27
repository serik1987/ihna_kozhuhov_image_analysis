# -*- coding: utf-8

import wx


class TwoMapSelector(wx.Dialog):

    __first_map = None
    __second_map = None

    __first_map_box = None
    __second_map_box = None

    __case = None

    def get_first_map(self):
        return self.__first_map

    def get_second_map(self):
        return self.__second_map

    def __init__(self, parent, case):
        data_names = parent.get_data_names()
        self.__case = case
        super().__init__(parent, title="Two maps selector", size=(500, 400))
        panel = wx.Panel(self)
        general_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        upper_sizer = wx.FlexGridSizer(2, 5, 5)
        upper_sizer.AddGrowableCol(0, 1)
        upper_sizer.AddGrowableCol(1, 1)
        upper_sizer.AddGrowableRow(1, 1)

        first_map_label = wx.StaticText(panel, label="First map")
        upper_sizer.Add(first_map_label, 1, wx.EXPAND)

        second_map_label = wx.StaticText(panel, label="Second map")
        upper_sizer.Add(second_map_label, 1, wx.EXPAND)

        self.__first_map_box = wx.ListBox(panel, choices=data_names, style=wx.LB_SINGLE | wx.LB_NEEDED_SB | wx.LB_SORT)
        upper_sizer.Add(self.__first_map_box, 1, wx.EXPAND)

        self.__second_map_box = wx.ListBox(panel, choices=data_names,
                                           style=wx.LB_SINGLE | wx.LB_NEEDED_SB | wx.LB_SORT)
        upper_sizer.Add(self.__second_map_box, 1, wx.EXPAND)

        main_sizer.Add(upper_sizer, 1, wx.EXPAND | wx.BOTTOM, 10)
        lower_sizer = wx.BoxSizer(wx.HORIZONTAL)

        ok = wx.Button(panel, label="Continue")
        ok.Bind(wx.EVT_BUTTON, lambda event: self.__continue())
        lower_sizer.Add(ok, 0, wx.RIGHT, 5)

        cancel = wx.Button(panel, label="Cancel")
        cancel.Bind(wx.EVT_BUTTON, lambda event: self.Close())
        lower_sizer.Add(cancel)

        main_sizer.Add(lower_sizer, 0, wx.ALIGN_CENTRE)
        general_sizer.Add(main_sizer, 1, wx.ALL | wx.EXPAND, 10)
        panel.SetSizer(general_sizer)
        self.Centre()

    def __continue(self):
        self.__first_map = self.__case.get_data(self.__first_map_box.GetStringSelection())
        self.__second_map = self.__case.get_data(self.__second_map_box.GetStringSelection())
        self.EndModal(wx.ID_OK)
