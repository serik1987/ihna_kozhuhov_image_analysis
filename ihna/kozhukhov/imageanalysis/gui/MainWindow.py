# -*- coding: utf-8

import wx


class MainWindow(wx.Frame):
    """
    This is the main window of the application
    """

    __cases_box = None
    __new_animal = None
    __delete_animal = None
    __animal_filter = None
    __specimen_box = None
    __conditions_box = None
    __recording_site_box = None
    __save_animal_info = None

    def __init__(self):
        super().__init__(None, title="Image Analysis", size=(800, 700),
                         style=wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX))
        panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.HORIZONTAL)
        main_layout = wx.BoxSizer(wx.HORIZONTAL)

        left_panel = wx.BoxSizer(wx.VERTICAL)
        left_panel_caption = wx.StaticText(panel, label="Case name", style=wx.ALIGN_LEFT)
        left_panel.Add(left_panel_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__cases_box = wx.ListBox(panel, size=(100, 300), style=wx.LB_SINGLE|wx.LB_NEEDED_SB|wx.LB_SORT)
        left_panel.Add(self.__cases_box, 0, wx.BOTTOM | wx.EXPAND, 5)
        left_button_panel = wx.BoxSizer(wx.HORIZONTAL)

        self.__new_animal = wx.Button(panel, label="New animal")
        left_button_panel.Add(self.__new_animal, 0, wx.RIGHT, 5)

        self.__delete_animal = wx.Button(panel, label="Delete animal")
        self.__delete_animal.Enable(False)
        left_button_panel.Add(self.__delete_animal, 0, wx.RIGHT, 5)

        self.__animal_filter = wx.Button(panel, label="Animal filter")
        left_button_panel.Add(self.__animal_filter, 0, 0, 0)

        left_panel.Add(left_button_panel, 0, wx.BOTTOM, 5)
        left_box = wx.StaticBox(panel, label="Animal info", style=wx.ALIGN_LEFT)
        left_panel.Add(left_box, 1, wx.EXPAND, 0)

        left_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        left_panel_content_sizer = wx.BoxSizer(wx.VERTICAL)
        specimen_box_caption = wx.StaticText(left_box, label="Specimen")
        left_panel_content_sizer.Add(specimen_box_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__specimen_box = wx.TextCtrl(left_box, value="")
        left_panel_content_sizer.Add(self.__specimen_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        conditions_caption = wx.StaticText(left_box, label="Conditions")
        left_panel_content_sizer.Add(conditions_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__conditions_box = wx.TextCtrl(left_box)
        left_panel_content_sizer.Add(self.__conditions_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        recording_site_label = wx.StaticText(left_box, label="Recording site")
        left_panel_content_sizer.Add(recording_site_label, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__recording_site_box = wx.TextCtrl(left_box)
        left_panel_content_sizer.Add(self.__recording_site_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__save_animal_info = wx.Button(left_box, label="Save animal info")
        left_panel_content_sizer.Add(self.__save_animal_info, 0, 0, 0)

        left_panel_sizer.Add(left_panel_content_sizer, 1, wx.ALL | wx.EXPAND, 5)
        left_box.SetSizer(left_panel_sizer)
        main_layout.Add(left_panel, 3, wx.RIGHT | wx.EXPAND, 5)
        middle_panel = wx.Panel(panel)
        middle_panel.SetBackgroundColour("blue")
        main_layout.Add(middle_panel, 3, wx.RIGHT | wx.EXPAND, 5)

        right_panel = wx.Panel(panel)
        right_panel.SetBackgroundColour("yellow")
        main_layout.Add(right_panel, 2, wx.EXPAND, 0)

        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        panel.SetSizer(general_layout)
        self.Centre(wx.BOTH)
