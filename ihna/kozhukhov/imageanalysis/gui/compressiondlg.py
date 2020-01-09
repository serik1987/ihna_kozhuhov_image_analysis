# -*- coding: utf-8

import wx


class CompressionDlg(wx.Dialog):
    """
    This dialog gets the compression options from the user I/O
    """

    __fail_on_target_exists_box = None
    __delete_after_process_box = None

    def __init__(self, parent, title, fail_on_target_exists_name, delete_after_process_name, button_label):
        super().__init__(parent, title=title, size=(800, 600))
        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)

        self.__fail_on_target_exists_box = wx.CheckBox(main_panel, label=fail_on_target_exists_name)
        main_layout.Add(self.__fail_on_target_exists_box, 0, wx.BOTTOM, 5)

        self.__delete_after_process_box = wx.CheckBox(main_panel, label=delete_after_process_name)
        main_layout.Add(self.__delete_after_process_box, 0, wx.BOTTOM, 15)

        buttons_panel = wx.BoxSizer(wx.HORIZONTAL)

        btn_ok = wx.Button(main_panel, label=button_label)
        self.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_OK), btn_ok)
        buttons_panel.Add(btn_ok, 0, wx.RIGHT, 5)

        btn_cancel = wx.Button(main_panel, label="Cancel")
        self.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_CANCEL), btn_cancel)
        buttons_panel.Add(btn_cancel)

        main_layout.Add(buttons_panel, 0, wx.ALIGN_CENTER, 0)
        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizerAndFit(general_layout)
        self.Centre()
        self.Fit()

    def is_fail_on_target_exists(self):
        return self.__fail_on_target_exists_box.GetValue()

    def is_delete_after_process(self):
        return self.__delete_after_process_box.GetValue()
