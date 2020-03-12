# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.gui.synchronization import synchronization_editors


class SynchronizationSelector(wx.StaticBoxSizer):
    """
    Represents the widgets that allows to select an appropriate synchronization and adjust its parameters
    """

    __parent = None
    __train = None
    __sync_editors = None
    __do_precise_box = None
    __harmonic_box = None

    def __init__(self, parent, train):
        super().__init__(wx.VERTICAL, parent, label="Synchronization")
        self.__train = train
        main_panel = wx.BoxSizer(wx.VERTICAL)
        self.__sync_editors = []

        for SyncEditor in synchronization_editors:
            editor = SyncEditor(parent, self.__train, self)
            self.__sync_editors.append(editor)
            space = 5
            main_panel.Add(editor, 0, wx.EXPAND | wx.BOTTOM, space)
        self.select()

        self.__do_precise_box = wx.CheckBox(parent, label="Do precise analysis")
        main_panel.Add(self.__do_precise_box, 0, wx.EXPAND | wx.BOTTOM, 5)

        harmonic_panel = wx.BoxSizer(wx.HORIZONTAL)
        harmonic_caption = wx.StaticText(parent, label="Harmonic")
        harmonic_panel.Add(harmonic_caption, wx.ALIGN_CENTER | wx.RIGHT, 5)

        self.__harmonic_box = wx.TextCtrl(parent, value="1.0")
        harmonic_panel.Add(self.__harmonic_box, wx.EXPAND)

        main_panel.Add(harmonic_panel, 0, wx.EXPAND)
        self.Add(main_panel, 0, wx.EXPAND | wx.ALL, 5)

    def close(self):
        self.__train = None
        self.__parent = None
        for sync_editor in self.__sync_editors:
            sync_editor.close()

    def select(self):
        for editor in self.__sync_editors:
            if editor.has_parameters() and editor.is_selected():
                editor.enable()
            if editor.has_parameters() and not editor.is_selected():
                editor.disable()

    def create_synchronization(self):
        for editor in self.__sync_editors:
            if editor.is_selected():
                sync = editor.create_synchronization(self.__train)
                sync.do_precise = self.__do_precise_box.IsChecked()
                try:
                    sync.harmonic = float(self.__harmonic_box.GetValue())
                except ValueError:
                    raise ValueError("Please, enter a correct value of the harmonic in the Synchronization section")
                return sync

    def get_options(self):
        for editor in self.__sync_editors:
            if editor.is_selected():
                return editor.get_options()
