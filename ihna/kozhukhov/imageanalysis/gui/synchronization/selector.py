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

    def __init__(self, parent, train):
        super().__init__(wx.VERTICAL, parent, label="Synchronization")
        self.__train = train
        main_panel = wx.BoxSizer(wx.VERTICAL)
        self.__sync_editors = []

        for SyncEditor in synchronization_editors:
            editor = SyncEditor(parent, self.__train, self)
            self.__sync_editors.append(editor)
            space = 5
            if len(self.__sync_editors) == len(synchronization_editors):
                space = 0
            main_panel.Add(editor, 0, wx.EXPAND | wx.BOTTOM, space)
        self.select()

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
