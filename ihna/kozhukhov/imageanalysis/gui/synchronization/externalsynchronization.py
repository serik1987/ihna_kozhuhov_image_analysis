# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.synchronization import ExternalSynchronization
from .synchronization import SynchronizationEditor


class ExternalSynchronizationEditor(SynchronizationEditor):
    """
    The editor provides a GUI for creating the ExternalSynchronization object and set all its necessary properties
    """

    __failed = False
    __channel_caption = None
    __channel_box = None
    __initial_cycle_set_box = None
    __initial_cycle_box = None
    __final_cycle_set_box = None
    __final_cycle_box = None

    def get_name(self):
        return "Synchronization by the external signal"

    def __init__(self, parent, train, selector):
        super().__init__(parent, train, selector)

    def put_controls(self):
        self.__failed = False

        try:

            if not self._train.is_opened:
                raise RuntimeError("Please, open the file train before creating the dialog")
            chans = self._train.synchronization_channel_number
            choices = [str(chan) for chan in range(chans)]

            controls = wx.FlexGridSizer(2, 5, 5)

            self.__channel_caption = wx.StaticText(self._parent, label="Select a synchronization channel")
            controls.Add(self.__channel_caption, 0, wx.ALIGN_CENTER_VERTICAL)

            self.__channel_box = wx.Choice(self._parent, choices=choices)
            self.__channel_box.SetSelection(0)
            controls.Add(self.__channel_box, 1, wx.EXPAND)

            self.__initial_cycle_set_box = wx.CheckBox(self._parent, label="Start analysis from cycle #")
            self.__initial_cycle_set_box.Bind(wx.EVT_CHECKBOX, lambda event: self.set_initial_cycle_enability())
            controls.Add(self.__initial_cycle_set_box, 0, wx.ALIGN_CENTER_VERTICAL)

            self.__initial_cycle_box = wx.TextCtrl(self._parent)
            self.__initial_cycle_box.Enable(False)
            controls.Add(self.__initial_cycle_box, 1, wx.EXPAND)

            self.__final_cycle_set_box = wx.CheckBox(self._parent, label="Finish analysis at cycle #")
            self.__final_cycle_set_box.Bind(wx.EVT_CHECKBOX, lambda event: self.set_final_cycle_enability())
            controls.Add(self.__final_cycle_set_box, 0, wx.ALIGN_CENTER_VERTICAL)

            self.__final_cycle_box = wx.TextCtrl(self._parent)
            self.__final_cycle_box.Enable(False)
            controls.Add(self.__final_cycle_box, 1, wx.EXPAND)

            controls.AddGrowableCol(1, 1)

            return controls

        except Exception as err:
            print("External synchronization is not possible now")
            print("Reason:", err.__class__.__name__)
            print("Comment:", str(err))
            self.__failed = True
            self._rb.Enable(False)
            return None

    def has_parameters(self):
        return not self.__failed

    def enable(self):
        self.__channel_box.Enable(True)
        self.__channel_caption.Enable(True)
        self.__initial_cycle_set_box.Enable(True)
        self.__final_cycle_set_box.Enable(True)
        self.set_initial_cycle_enability()
        self.set_final_cycle_enability()

    def disable(self):
        self.__channel_caption.Enable(False)
        self.__channel_box.Enable(False)
        self.__initial_cycle_set_box.Enable(False)
        self.__initial_cycle_box.Enable(False)
        self.__final_cycle_set_box.Enable(False)
        self.__final_cycle_box.Enable(False)

    def set_initial_cycle_enability(self):
        if self.__initial_cycle_set_box.IsChecked():
            self.__initial_cycle_box.Enable(True)
        else:
            self.__initial_cycle_box.Enable(False)
            self.__initial_cycle_box.SetValue("")

    def set_final_cycle_enability(self):
        if self.__final_cycle_set_box.IsChecked():
            self.__final_cycle_box.Enable(True)
        else:
            self.__final_cycle_box.Enable(False)
            self.__final_cycle_box.SetValue("")

    def create_synchronization(self, train):
        sync = ExternalSynchronization(train)
        sync.channel_number = self.__channel_box.GetCurrentSelection()
        if self.__initial_cycle_set_box.IsChecked():
            try:
                sync.initial_cycle = int(self.__initial_cycle_box.GetValue())
            except ValueError:
                raise ValueError("Please, set the correct value of the initial cycle or don't check this box")
        if self.__final_cycle_set_box.IsChecked():
            try:
                sync.final_cycle = int(self.__final_cycle_box.GetValue())
            except ValueError:
                raise ValueError("Please, set the correct value of the final cycle or don't check this box")
        return sync
