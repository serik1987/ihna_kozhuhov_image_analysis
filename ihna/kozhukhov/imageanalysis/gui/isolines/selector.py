# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.sourcefiles import StreamFileTrain
from ihna.kozhukhov.imageanalysis.gui.isolines import editor_list


class IsolineSelector(wx.StaticBoxSizer):
    """
    Provides a widget to select an appropriate Isoline and create its instance
    """

    __parent = None
    __train = None
    __editors = None

    def __init__(self, parent, train: StreamFileTrain):
        super().__init__(wx.VERTICAL, parent, label="Isoline remove")
        self.__parent = parent
        self.__train = train

        main_panel = wx.BoxSizer(wx.VERTICAL)

        self.__editors = []
        for IsolineEditor in editor_list:
            editor = IsolineEditor(parent, train, self)
            self.__editors.append(editor)
            border = 5
            main_panel.Add(editor, 0, wx.BOTTOM | wx.EXPAND, border)
        self.select()

        self.Add(main_panel, 1, wx.ALL | wx.EXPAND, 5)

    def close(self):
        self.__parent = None
        self.__train = None

        for editor in self.__editors:
            editor.close()

    def select(self):
        for editor in self.__editors:
            if editor.is_selected() and editor.has_properties():
                editor.enable()
            if not editor.is_selected() and editor.has_properties():
                editor.disable()

    def create_isoline(self, sync, train=None):
        if train is None:
            train = self.__train
        for editor in self.__editors:
            if editor.is_selected():
                return editor.create_isoline(train, sync)

    def get_options(self):
        for editor in self.__editors:
            if editor.is_selected():
                return editor.get_options()

