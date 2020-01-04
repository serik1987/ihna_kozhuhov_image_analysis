# -*- coding: utf-8

import wx


class DefineSimpleRoiDlg(wx.Dialog):
    """
    This dialog is to define the simple ROI
    """

    __vessel_map_widget = None
    __complex_map_widget = None
    __amplitude_map_widget = None
    __phase_map_widget = None

    __name_box = None

    __left_box = None
    __right_box = None
    __top_box = None
    __bottom_box = None
    __width_box = None
    __height_box = None
    __area_box = None

    __current_border = "left"

    def __init__(self, parent, fullname, vessel_map=None, amplitude_map=None, phase_map=None):
        super().__init__(parent, title="Define simple ROI: " + fullname, size=(830, 620))
        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.HORIZONTAL)

        maps = self.__create_maps(main_panel, vessel_map, amplitude_map, phase_map)
        main_layout.Add(maps, 6, wx.RIGHT | wx.EXPAND, 10)

        controls = self.__create_controls(main_panel)
        main_layout.Add(controls, 2, wx.EXPAND)

        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizer(general_layout)
        self.Centre()

    def __create_maps(self, parent, vessel_map, amplitude_map, phase_map):
        vessel_map = (vessel_map - vessel_map.min()) / (vessel_map.max() - vessel_map.min())

        maps = wx.GridSizer(2, 5, 5)

        print("PY Drawing vessel map")
        self.__vessel_map_widget = DefineSimpleRoiPanel(parent, vessel_map, "gray", "red")
        maps.Add(self.__vessel_map_widget, 1, wx.EXPAND)

        print("PY Drawing complex map")
        self.__complex_map_widget = DefineSimpleRoiPanel(parent, vessel_map, "jet", "green")
        maps.Add(self.__complex_map_widget, 1, wx.EXPAND)

        print("PY Drawing amplitude map")
        self.__amplitude_map_widget = DefineSimpleRoiPanel(parent, vessel_map, "gray", "blue")
        maps.Add(self.__amplitude_map_widget, 1, wx.EXPAND)

        print("PY Drawing phase map")
        self.__phase_map_widget = DefineSimpleRoiPanel(parent, vessel_map, "jet", "yellow")
        maps.Add(self.__phase_map_widget, 1, wx.EXPAND)

        print("PY Drawing completed")

        return maps

    def __set_radio(self, value):
        self.__current_border = value

    def finalize_roi_definition(self):
        print("PY Finalize ROI definition")

    def __create_controls(self, parent):
        controls = wx.BoxSizer(wx.VERTICAL)

        name_prompt = wx.StaticText(parent, label="ROI name (*)")
        controls.Add(name_prompt, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__name_box = wx.TextCtrl(parent)
        controls.Add(self.__name_box, 0, wx.EXPAND | wx.BOTTOM, 25)

        left_radio = wx.RadioButton(parent, label="Left border", style=wx.RB_GROUP)
        self.Bind(wx.EVT_RADIOBUTTON, lambda event: self.__set_radio("left"), left_radio)
        controls.Add(left_radio, 0, wx.BOTTOM | wx.EXPAND, 5)

        right_radio = wx.RadioButton(parent, label="Right border")
        self.Bind(wx.EVT_RADIOBUTTON, lambda event: self.__set_radio("right"), right_radio)
        controls.Add(right_radio, 0, wx.BOTTOM | wx.EXPAND, 5)

        top_radio = wx.RadioButton(parent, label="Top border")
        self.Bind(wx.EVT_RADIOBUTTON, lambda event: self.__set_radio("top"), top_radio)
        controls.Add(top_radio, 0, wx.BOTTOM | wx.EXPAND, 5)

        bottom_radio = wx.RadioButton(parent, label="Bottom border")
        self.Bind(wx.EVT_RADIOBUTTON, lambda event: self.__set_radio("bottom"), bottom_radio)
        controls.Add(bottom_radio, 0, wx.BOTTOM | wx.EXPAND, 25)

        subcontrols = wx.GridSizer(2, 5, 5)

        self.__left_box = wx.StaticText(parent, label="left = 1024")
        subcontrols.Add(self.__left_box, 0, wx.EXPAND)

        self.__right_box = wx.StaticText(parent, label="right = 1024")
        subcontrols.Add(self.__right_box, 0, wx.EXPAND)

        self.__bottom_box = wx.StaticText(parent, label="bottom = 1024")
        subcontrols.Add(self.__bottom_box, 0, wx.EXPAND)

        self.__top_box = wx.StaticText(parent, label="top = 1024")
        subcontrols.Add(self.__top_box, 0, wx.EXPAND)

        self.__width_box = wx.StaticText(parent, label="width = 1024")
        subcontrols.Add(self.__width_box, 0, wx.EXPAND)

        self.__height_box = wx.StaticText(parent, label="height = 1024")
        subcontrols.Add(self.__height_box, 0, wx.EXPAND)

        controls.Add(subcontrols, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__area_box = wx.StaticText(parent, label="area = {0}".format(1024**2), style=wx.ALIGN_CENTRE_HORIZONTAL)
        controls.Add(self.__area_box, 1, wx.BOTTOM | wx.EXPAND, 50)

        buttons = wx.BoxSizer(wx.HORIZONTAL)

        ok = wx.Button(parent, label="OK")
        self.Bind(wx.EVT_BUTTON, lambda event: self.finalize_roi_definition(), ok)
        buttons.Add(ok, 0, wx.RIGHT, 5)

        cancel = wx.Button(parent, label="Cancel")
        self.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_CANCEL), cancel)
        buttons.Add(cancel)

        controls.Add(buttons, 0, wx.ALIGN_CENTER)
        return controls


class DefineSimpleRoiPanel(wx.Panel):
    """
    Represents a panel that plots single maps to draw the ROI
    """

    def __init__(self, parent, map, colormap, placeholder):
        """
        Arguments:
            parent - the parent widget
            map - numpy array of complex values, which amplitudes shall be in range [0, 1]
            colormap - matplotlib's colormap to be drawn, 'gray' and 'hsv parameters only are supported
            placeholder - what shall be placed if the map is None
        """
        super().__init__(parent)
        self.SetBackgroundColour(placeholder)

        if map is not None:
            print(map)
