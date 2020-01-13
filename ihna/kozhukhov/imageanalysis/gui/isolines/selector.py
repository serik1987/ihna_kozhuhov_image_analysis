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

    def __init__(self, parent, train: StreamFileTrain):
        super().__init__(wx.VERTICAL, parent, label="Isoline remove")
        self.__parent = parent
        self.__train = train
