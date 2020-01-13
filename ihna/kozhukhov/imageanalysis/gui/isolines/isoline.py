# -*- coding: utf-8

import wx


class IsolineEditor(wx.BoxSizer):
    """
    This is the base class for GUI tools that allow to select an certain Isoline and set its properties
    """

    _parent = None
    _train = None
    _sync = None

    def get_name(self):
        """
        Returns a string that will be drawn at the right of the radio button which selects this isoline
        """
        raise NotImplementedError("IsolineEditor is abstract. Use any of its derived classes")

    def __init__(self, parent, train):
        """
        Isoline initialization

        Arguments:
            parent - the parent window
            train - train to the isoline
        """
        super().__init__(wx.VERTICAL)
        self._parent = parent
        self._train = train
        self._sync = None
        print("Isoline name:", self.get_name())
