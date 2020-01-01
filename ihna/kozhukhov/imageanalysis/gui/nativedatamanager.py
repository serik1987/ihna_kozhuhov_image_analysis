# -*- coding: utf-8

import os.path
import wx
import ihna.kozhukhov.imageanalysis.sourcefiles as sfiles
from .chunk import ChunkViewer


class NativeDataManager(wx.Dialog):
    """
    This class is used to open the native data manager

    Usage:
    NativeDataManager(parent, case).ShowModal()
    where:
    parent - the parent window (instance of wx.Frame or wx.Dialog)
    case - an instance of ihna.kozhukhov.imageanalysis.manifest.Case
    """

    __case = None
    __train = None

    def __init__(self, parent, case):
        self.__case = case
        pathname = self.__case['pathname']
        if self.__case.compressed_data_files_exist():
            filename = self.__case['compressed_data_files'][0]
            fullname = os.path.join(pathname, filename)
            self.__train = sfiles.CompressedFileTrain(fullname, "traverse")
        elif self.__case.native_data_files_exist():
            filename = self.__case['native_data_files'][0]
            fullname = os.path.join(pathname, filename)
            self.__train = sfiles.StreamFileTrain(fullname, "traverse")
        else:
            raise RuntimeError("Error in opening native or compressed files")
        self.__train.open()
        title = "Native data manager: " + self.__case['short_name']
        super().__init__(parent, title=title, size=(800, 600))
        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.HORIZONTAL)

        left_panel = self.__create_left_panel(main_panel)
        main_layout.Add(left_panel, 6, wx.RIGHT | wx.EXPAND, 5)

        right_panel = self.__create_right_panel(main_panel)
        main_layout.Add(right_panel, 2, wx.EXPAND, 0)

        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizer(general_layout)
        self.Centre()

    def __create_left_panel(self, main_panel):
        left_panel = wx.Notebook(main_panel)

        general_properties_page = wx.Panel(left_panel)
        general_properties_page.SetBackgroundColour("red")
        left_panel.AddPage(general_properties_page, "General Properties")

        for file in self.__train:
            break

        for chunk in file.isoi:
            if chunk['id'] != "SOFT":
                viewer = ChunkViewer.new_viewer(left_panel, chunk)
                left_panel.AddPage(viewer, viewer.get_title())

        return left_panel

    def __create_right_panel(self, main_panel):
        right_panel = wx.Notebook(main_panel)
        right_panel.SetBackgroundColour("blue")

        return right_panel

    def close(self):
        """
        Closes the file opened by this dialog
        """
        self.__train.close()
