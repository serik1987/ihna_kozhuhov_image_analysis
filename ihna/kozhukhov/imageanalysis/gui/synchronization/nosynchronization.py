# -*- coding: utf-8

import wx
from .synchronization import SynchronizationEditor
from ihna.kozhukhov.imageanalysis.synchronization import NoSynchronization


class NoSynchronizationEditor(SynchronizationEditor):
    """
    Represents the set of widgets that allows to create an instance of NoSynchronization class
    """

    __isInitFrameBox = None
    __isFinalFrameBox = None
    __initFrameBox = None
    __finalFrameBox = None

    def get_name(self):
        return "No synchronization"

    def is_first(self):
        return True

    def has_parameters(self):
        return True

    def __init__(self, parent, train, selector):
        super().__init__(parent, train, selector)

    def put_controls(self):
        controls = wx.FlexGridSizer(2, 5, 5)

        self.__isInitFrameBox = wx.CheckBox(self._parent, label="Start analysis from frame #")
        self.__isInitFrameBox.Bind(wx.EVT_CHECKBOX, lambda event: self.change_init_frame_enability())
        controls.Add(self.__isInitFrameBox, 0, wx.ALIGN_CENTER_VERTICAL)

        self.__initFrameBox = wx.TextCtrl(self._parent)
        self.__initFrameBox.Enable(False)
        controls.Add(self.__initFrameBox)

        self.__isFinalFrameBox = wx.CheckBox(self._parent, label="Finish analysis by frame #")
        self.__isFinalFrameBox.Bind(wx.EVT_CHECKBOX, lambda event: self.change_final_frame_enability())
        controls.Add(self.__isFinalFrameBox, 0, wx.ALIGN_CENTER_VERTICAL)

        self.__finalFrameBox = wx.TextCtrl(self._parent)
        self.__finalFrameBox.Enable(False)
        controls.Add(self.__finalFrameBox)

        return controls

    def change_init_frame_enability(self):
        if self.__isInitFrameBox.IsChecked():
            self.__initFrameBox.Enable(True)
        else:
            self.__initFrameBox.Enable(False)
            self.__initFrameBox.SetValue("")

    def change_final_frame_enability(self):
        if self.__isFinalFrameBox.IsChecked():
            self.__finalFrameBox.Enable(True)
        else:
            self.__finalFrameBox.Enable(False)
            self.__finalFrameBox.SetValue("")

    def enable(self):
        self.__isInitFrameBox.Enable(True)
        self.change_init_frame_enability()
        self.__isFinalFrameBox.Enable(True)
        self.change_final_frame_enability()

    def disable(self):
        self.__isInitFrameBox.Enable(False)
        self.__initFrameBox.Enable(False)
        self.__isFinalFrameBox.Enable(False)
        self.__finalFrameBox.Enable(False)

    def is_selected(self):
        return self._rb.GetValue()

    def create_synchronization(self, train):
        sync = NoSynchronization(train)
        return sync
