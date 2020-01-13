# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.gui.synchronization import synchronization_editors


class SynchronizationSelector(wx.StaticBoxSizer):
    """
    Represents the widgets that allows to select an appropriate synchronization and adjust its parameters
    """

    __parent = None
    __train = None

    def __init__(self, parent, train):
        super().__init__(wx.VERTICAL, parent, label="Synchronization")
        self.__parent = parent
        self.__train = train
