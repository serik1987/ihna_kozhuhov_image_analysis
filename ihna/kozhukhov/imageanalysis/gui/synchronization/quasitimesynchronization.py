# -*- coding: utf-8


import wx
from ihna.kozhukhov.imageanalysis.synchronization import QuasiTimeSynchronization
from .synchronization import SynchronizationEditor


class QuasiTimeSynchronizationEditor(SynchronizationEditor):
    """
    Represents an editor that allows to create QuasiTimeSynchronization and set all its parameters
    """

    __stimulus_period_caption = None
    __stimulus_period_box = None
    __initial_cycle_set_box = None
    __initial_cycle_box = None
    __final_cycle_set_box = None
    __final_cycle_box = None

    def get_name(self):
        return "Quasi-time synchronization"

    def __init__(self, parent, train, selector):
        super().__init__(parent, train, selector)

    def has_parameters(self):
        return True

    def put_controls(self):
        controls = wx.FlexGridSizer(2, 5, 5)

        self.__stimulus_period_caption = wx.StaticText(self._parent, label="Stimulus period, ms")
        controls.Add(self.__stimulus_period_caption, 0, wx.ALIGN_CENTER_VERTICAL)

        self.__stimulus_period_box = wx.TextCtrl(self._parent)
        controls.Add(self.__stimulus_period_box, 1, wx.EXPAND)

        self.__initial_cycle_set_box = wx.CheckBox(self._parent, label="Start analysis from cycle #")
        self.__initial_cycle_set_box.Bind(wx.EVT_CHECKBOX, lambda event: self.set_initial_frame_enability())
        controls.Add(self.__initial_cycle_set_box, 0, wx.ALIGN_CENTER_VERTICAL)

        self.__initial_cycle_box = wx.TextCtrl(self._parent)
        self.__initial_cycle_box.Enable(False)
        controls.Add(self.__initial_cycle_box, 1, wx.EXPAND)

        self.__final_cycle_set_box = wx.CheckBox(self._parent, label="Finish analysis at the cycle #")
        self.__final_cycle_set_box.Bind(wx.EVT_CHECKBOX, lambda event: self.set_final_frame_enability())
        controls.Add(self.__final_cycle_set_box, 0, wx.ALIGN_CENTER_VERTICAL)

        self.__final_cycle_box = wx.TextCtrl(self._parent)
        self.__final_cycle_box.Enable(False)
        controls.Add(self.__final_cycle_box, 1, wx.EXPAND)

        return controls

    def enable(self):
        self.__stimulus_period_caption.Enable(True)
        self.__stimulus_period_box.Enable(True)
        self.__initial_cycle_set_box.Enable(True)
        self.__final_cycle_set_box.Enable(True)
        self.set_initial_frame_enability()
        self.set_final_frame_enability()

    def disable(self):
        self.__stimulus_period_caption.Enable(False)
        self.__stimulus_period_box.Enable(False)
        self.__initial_cycle_set_box.Enable(False)
        self.__final_cycle_set_box.Enable(False)
        self.__initial_cycle_box.Enable(False)
        self.__final_cycle_box.Enable(False)

    def set_initial_frame_enability(self):
        if self.__initial_cycle_set_box.IsChecked():
            self.__initial_cycle_box.Enable(True)
        else:
            self.__initial_cycle_box.Enable(False)
            self.__initial_cycle_box.SetValue("")

    def set_final_frame_enability(self):
        if self.__final_cycle_set_box.IsChecked():
            self.__final_cycle_box.Enable(True)
        else:
            self.__final_cycle_box.Enable(False)
            self.__final_cycle_box.SetValue("")

    def create_synchronization(self, train):
        sync = QuasiTimeSynchronization(train)
        return sync
