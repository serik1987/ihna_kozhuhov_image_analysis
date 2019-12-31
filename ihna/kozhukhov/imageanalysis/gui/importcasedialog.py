# -*- coding: utf-8

import wx


class ImportCaseDialog(wx.Dialog):
    """
    This is the main dialog that may be used for the case import
    """

    __download_box = None
    __link_box = None
    __sign_field = 50
    __result = None
    ID_DOWNLOAD = wx.ID_YES
    ID_LINK = wx.ID_NO

    def __add_sign(self, panel, sign, message):
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        if sign == "plus":
            label = "Plus:"
            color = "green"
        elif sign == "minus":
            label = "Minus:"
            color = "red"
        else:
            raise ValueError("2nd argument shall be either 'plus' or 'minus'")
        sign_box = wx.StaticText(panel, label=label)
        sign_box.SetForegroundColour(color)
        size = sign_box.GetSize()
        sign_box.SetSizeHints(self.__sign_field, size.height)
        sizer.Add(sign_box, 0, 0, 0)

        message_box = wx.StaticText(panel, label=message)
        sizer.Add(message_box, 0, 0, 0)

        return sizer

    def __init__(self, parent):
        super().__init__(parent, title="Import cases")
        panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)

        self.__download_box = wx.RadioButton(panel, label="Download from the external storage",
                                             style=wx.RB_GROUP)
        main_layout.Add(self.__download_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        plus = self.__add_sign(panel, "plus", "The data will be available after you unplug your external storage")
        main_layout.Add(plus, 0, wx.BOTTOM, 5)

        plus = self.__add_sign(panel, "plus", "Compatible with any operating system")
        main_layout.Add(plus, 0, wx.BOTTOM, 5)

        minus = self.__add_sign(panel, "minus", "The import will take too long time")
        main_layout.Add(minus, 0, wx.BOTTOM, 5)

        minus = self.__add_sign(panel, "minus", "The memory on your PC may not be enough")
        main_layout.Add(minus, 0, wx.BOTTOM, 5)

        self.__link_box = wx.RadioButton(panel, label="Create a symbolic link")
        main_layout.Add(self.__link_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        plus = self.__add_sign(panel, "plus", "The import takes no time")
        main_layout.Add(plus, 0, wx.BOTTOM, 5)

        plus = self.__add_sign(panel, "plus", "The import doesn't require any memory on your internal storage")
        main_layout.Add(plus, 0, wx.BOTTOM, 5)

        minus = self.__add_sign(panel, "minus", "The data will not be available without the external storage")
        main_layout.Add(minus, 0, wx.BOTTOM, 5)

        minus = self.__add_sign(panel, "minus", "May not be supported in your operating system")
        main_layout.Add(minus, 0, wx.BOTTOM, 5)

        button_panel = wx.BoxSizer(wx.HORIZONTAL)

        ok = wx.Button(panel, label="OK")
        self.Bind(wx.EVT_BUTTON, lambda event: self.__ok(), ok)
        button_panel.Add(ok, 0, wx.RIGHT, 5)

        cancel = wx.Button(panel, label="Cancel")
        self.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_CANCEL), cancel)
        button_panel.Add(cancel, 0, 0, 0)

        main_layout.Add(button_panel, 0, wx.ALIGN_CENTER, 0)
        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        panel.SetSizerAndFit(general_layout)
        self.Centre()
        self.Fit()

    def __ok(self):
        if self.__download_box.GetValue():
            self.__result = self.ID_DOWNLOAD
            self.EndModal(self.ID_DOWNLOAD)
        if self.__link_box.GetValue():
            self.__result = self.ID_LINK
            self.EndModal(self.ID_LINK)

    def get_result(self):
        return self.__result
