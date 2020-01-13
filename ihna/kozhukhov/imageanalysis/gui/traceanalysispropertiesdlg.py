# -*- coding: utf-8

import wx
from .synchronization.selector import SynchronizationSelector


class TraceAnalysisPropertiesDlg(wx.Dialog):
    """
    Provides a convenient dialog for setting trace analysis properties
    """

    __train = None
    __parent = None
    __sync_selector = None
    __isoline_selector = None
    __channel_selector = None

    def __init__(self, parent, train):
        super().__init__(parent, title="Trace analysis properties", size=(800, 600))
        self.__parent = parent
        self.__train = train

        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)

        upper_panel = self.__init_upper_panel(main_panel)
        main_layout.Add(upper_panel, 0, wx.BOTTOM, 10)

        middle_panel = self.__init_middle_panel(main_panel)
        main_layout.Add(middle_panel, 0, wx.BOTTOM, 10)

        lower_panel = self.__init_lower_panel(main_panel)
        main_layout.Add(lower_panel, 0, wx.ALIGN_CENTER)

        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizerAndFit(general_layout)
        self.Centre()
        self.Fit()

    def __init_upper_panel(self, parent):
        upper_panel = wx.BoxSizer(wx.HORIZONTAL)

        self.__sync_selector = SynchronizationSelector(parent, self.__train)
        upper_panel.Add(self.__sync_selector, 0, wx.RIGHT, 10)

        upper_right_panel = wx.BoxSizer(wx.VERTICAL)

        self.__isoline_selector = wx.Panel(parent, size=(300, 300))
        self.__isoline_selector.SetBackgroundColour("black")
        upper_right_panel.Add(self.__isoline_selector, 0, wx.BOTTOM | wx.EXPAND, 10)

        self.__channel_selector = wx.Panel(parent, size=(200, 100))
        self.__channel_selector.SetBackgroundColour("red")
        upper_right_panel.Add(self.__channel_selector, 0, wx.EXPAND)

        upper_panel.Add(upper_right_panel, 0)
        return upper_panel

    def __init_middle_panel(self, parent):
        middle_panel = wx.Panel(parent, size=(300, 50))
        middle_panel.SetBackgroundColour("blue")

        return middle_panel

    def __init_lower_panel(self, parent):
        lower_panel = wx.Panel(parent, size=(100, 50))
        lower_panel.SetBackgroundColour("orange")

        return lower_panel

    def close(self):
        """
        Deletes the file train that is associated with the dialog
        """
        del self.__train
        self.__train = None
        del self.__parent
        self.__parent = None

        self.__sync_selector.close()
