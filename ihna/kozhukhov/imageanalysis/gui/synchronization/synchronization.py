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

    def get_name(self):
        """
        Returns the synchronization name. The name will be shown at the right of an appropriate radio button
        """
        raise NotImplementedError("SynchronizationEditor is a base class. Use any of its derived class")

    def __init__(self, parent, train):
        self._parent = parent
        self._train = train
        super().__init__(wx.VERTICAL)
        print("PY Synchronization name: ", self.get_name())
