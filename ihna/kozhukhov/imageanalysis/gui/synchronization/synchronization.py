# -*- coding: utf-8

import wx


class SynchronizationEditor(wx.BoxSizer):
    """
    This is the base class for synchronization editor. The synchronization editor performs two functions:
    1) The editor places necessary widgets on you wx.Frame/wx.Dialog including:
    1.1) Radio button that will be used to select a certain Synchronization.
    1.2) Widgets to adjust the Synchronization properties
    2) The editor contains create() method that will create a synchronization object and adjusts all its parameters
    based on user selection/input
    """

    _parent = None
    _train = None
    _rb = None

    def get_name(self):
        """
        Returns the synchronization name. The name will be shown at the right of an appropriate radio button
        """
        raise NotImplementedError("SynchronizationEditor is a base class. Use any of its derived class")

    def is_first(self):
        return False

    def has_parameters(self):
        return False

    def put_controls(self):
        controls = wx.Panel(self._parent, size=(100, 100))
        controls.SetBackgroundColour("green")
        return controls

    def __init__(self, parent, train, selector):
        super().__init__(wx.VERTICAL)
        self._parent = parent
        self._train = train

        style = 0
        if self.is_first():
            style |= wx.RB_GROUP

        self._rb = wx.RadioButton(parent, label=self.get_name(), style=style)
        self._rb.Bind(wx.EVT_RADIOBUTTON, lambda event: selector.select())
        self.Add(self._rb)

        if self.has_parameters():
            main_panel = wx.BoxSizer(wx.HORIZONTAL)
            controls = self.put_controls()
            if controls is not None:
                main_panel.Add(controls, 1, wx.LEFT | wx.EXPAND, 20)
            self.Add(main_panel, 0, wx.TOP | wx.EXPAND, 5)

    def close(self):
        self._train = None
        self._parent = None

    def enable(self):
        raise NotImplementedError("enable()")

    def disable(self):
        raise NotImplementedError("disable()")

    def is_selected(self):
        return self._rb.GetValue()

    def get_synchronization(self):
        raise NotImplementedError("Synchronization")
