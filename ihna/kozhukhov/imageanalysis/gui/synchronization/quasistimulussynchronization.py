# -*- coding: utf-8

import wx
from .synchronization import SynchronizationEditor


class QuasiStimulusSynchronizationEditor(SynchronizationEditor):
    """
    Provides a GUI for creating QuasiStimulusSynchronization and setting its properties
    """

    __stimulusPeriodCaption = None
    __stimulusPeriodBox = None
    __initialCycleSetBox = None
    __finalCycleSetBox = None
    __initialCycleBox = None
    __finalCycleBox = None

    def get_name(self):
        return "Quasi-stimulus synchronization"

    def __init__(self, parent, train, selector):
        super().__init__(parent, train, selector)

    def has_parameters(self):
        return True

    def put_controls(self):
        controls = wx.FlexGridSizer(2, 5, 5)

        self.__stimulusPeriodCaption = wx.StaticText(self._parent, label="Stimulus period, frames")
        controls.Add(self.__stimulusPeriodCaption, 0, wx.ALIGN_CENTER_VERTICAL)

        self.__stimulusPeriodBox = wx.TextCtrl(self._parent)
        controls.Add(self.__stimulusPeriodBox, 1, wx.EXPAND)

        self.__initialCycleSetBox = wx.CheckBox(self._parent, label="Start analysis from cycle #")
        self.__initialCycleSetBox.Bind(wx.EVT_CHECKBOX, lambda event: self.set_initial_frame_enability())
        controls.Add(self.__initialCycleSetBox, 0, wx.ALIGN_CENTER_VERTICAL)

        self.__initialCycleBox = wx.TextCtrl(self._parent)
        self.__initialCycleBox.Enable(False)
        controls.Add(self.__initialCycleBox, 1, wx.EXPAND)

        self.__finalCycleSetBox = wx.CheckBox(self._parent, label="Finish analysis at the cycle #")
        self.__finalCycleSetBox.Bind(wx.EVT_CHECKBOX, lambda event: self.set_final_frame_enability())
        controls.Add(self.__finalCycleSetBox, 0, wx.ALIGN_CENTER_VERTICAL)

        self.__finalCycleBox = wx.TextCtrl(self._parent)
        self.__finalCycleBox.Enable(False)
        controls.Add(self.__finalCycleBox, 1, wx.EXPAND)

        return controls

    def enable(self):
        self.__stimulusPeriodCaption.Enable(True)
        self.__stimulusPeriodBox.Enable(True)
        self.__initialCycleSetBox.Enable(True)
        self.__finalCycleSetBox.Enable(True)
        self.set_initial_frame_enability()
        self.set_final_frame_enability()

    def disable(self):
        self.__stimulusPeriodCaption.Enable(False)
        self.__stimulusPeriodBox.Enable(False)
        self.__initialCycleSetBox.Enable(False)
        self.__finalCycleSetBox.Enable(False)
        self.__initialCycleBox.Enable(False)
        self.__finalCycleBox.Enable(False)

    def set_initial_frame_enability(self):
        if self.__initialCycleSetBox.IsChecked():
            self.__initialCycleBox.Enable(True)
        else:
            self.__initialCycleBox.Enable(False)
            self.__initialCycleBox.SetValue("")

    def set_final_frame_enability(self):
        if self.__finalCycleSetBox.IsChecked():
            self.__finalCycleBox.Enable(True)
        else:
            self.__finalCycleBox.Enable(False)
            self.__finalCycleBox.SetValue("")
