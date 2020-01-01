# -*- coding: utf-8

import wx


class ChunkViewer(wx.Panel):
    """
    This is the base class for all chunk tabs. Also, if the chunk is unknown or non-recognized,
    such a class corresponds to the chunk

    Initialization:
        viewer = ChunkViewer(parent, chunk)
    where parent is the parent WX window (wx.Notebook instance) and
    chunk is an instance of ihna.kozhukhov.imageanalysis.sourcefiles.Chunk object
    """

    __chunk = None

    def __init__(self, parent, chunk):
        self.__chunk = chunk
        super().__init__(parent)
        self.SetBackgroundColour("green")

    def get_title(self):
        """
        Returns the chunk title (to be substituted to wx.Notebook.AddPage arguments)
        """
        return self.__chunk['id']

    @staticmethod
    def new_viewer(parent, chunk):
        return ChunkViewer(parent, chunk)
