# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.sourcefiles import StreamFileTrain


class IsolineEditor(wx.BoxSizer):
    """
    This is the base class for GUI tools that allow to select an certain Isoline and set its properties
    """

    _parent = None
    _train = None
    _sync = None
    _rb = None

    def get_name(self) -> str:
        """
        Returns a string that will be drawn at the right of the radio button which selects this isoline
        """
        raise NotImplementedError("IsolineEditor is abstract. Use any of its derived classes")

    def __init__(self, parent, train: StreamFileTrain, selector):
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

        if self.is_first():
            style = wx.RB_GROUP
        else:
            style = 0

        self._rb = wx.RadioButton(parent, label=self.get_name(), style=style)
        self._rb.Bind(wx.EVT_RADIOBUTTON, lambda event: selector.select())
        self.Add(self._rb, 0, wx.EXPAND)

        if self.has_properties():
            controls_panel = wx.BoxSizer(wx.HORIZONTAL)
            controls = self.put_properties()
            controls_panel.Add(controls, 1, wx.LEFT | wx.EXPAND, 20)
            self.Add(controls_panel, 0, wx.TOP | wx.EXPAND, 5)

    def is_first(self):
        return False

    def close(self):
        self._parent = None
        self._train = None
        self._sync = None
        self._rb = None

    def has_properties(self):
        return False

    def put_properties(self):
        controls = wx.Panel(self._parent, size=(100, 100))
        controls.SetBackgroundColour("black")

        return controls

    def is_selected(self):
        return self._rb.GetValue()

    def enable(self):
        raise NotImplementedError("enable()")

    def disable(self):
        raise NotImplementedError("disable()")
