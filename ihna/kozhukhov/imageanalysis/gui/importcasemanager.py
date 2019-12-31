# -*- coding: utf-8

import shutil
import os
import os.path
import wx
from wx.lib.scrolledpanel import ScrolledPanel


class ImportCaseManager(wx.Dialog):
    """
    Represents the Import case manager and provides the routines for file copying and link creating
    """

    __valid_files = None
    __invalid_files = None
    __valid_file_editors = None
    __invalid_file_editors = None
    __do_links = None
    __import_button = None
    __close_button = None
    __cancel_button = None
    __destination = None
    __panel = None

    def __init__(self, parent, valid_files, invalid_files, do_links, destination):
        self.__valid_files = valid_files
        self.__invalid_files = invalid_files
        self.__valid_file_editors = []
        self.__invalid_file_editors = []
        self.__do_links = do_links
        self.__destination = destination

        super().__init__(parent, title="Import case manager", size=(800, 600))
        panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)
        scroll_panel = ScrolledPanel(panel, size=(800, 500))
        scroll_panel.SetupScrolling(False, True)
        scroll_panel_main_layout = wx.BoxSizer(wx.VERTICAL)

        for valid_file in valid_files:
            filename = valid_file['filename']
            filetype = valid_file['filetype']
            try:
                tail_files = valid_file['tail_files']
            except KeyError:
                tail_files = None
            pan = ValidFileEditor(scroll_panel, filename, filetype, tail_files)
            self.__valid_file_editors.append(pan)
            scroll_panel_main_layout.Add(pan.get_layout(), 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
            pan.set_ready()

        for invalid_file in invalid_files:
            pan = InvalidFileEditor(scroll_panel, invalid_file)
            self.__invalid_file_editors.append(pan)
            scroll_panel_main_layout.Add(pan.get_layout(), 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)

        scroll_panel.SetSizer(scroll_panel_main_layout)
        main_layout.Add(scroll_panel, 0, wx.EXPAND | wx.BOTTOM, 5)
        button_panel = wx.BoxSizer(wx.HORIZONTAL)

        self.__import_button = wx.Button(panel, label="Import")
        self.Bind(wx.EVT_BUTTON, lambda event: self.do_import(), self.__import_button)
        button_panel.Add(self.__import_button, 0, wx.RIGHT, 5)

        self.__cancel_button = wx.Button(panel, label="Cancel")
        self.Bind(wx.EVT_BUTTON, lambda event: self.do_cancel(), self.__cancel_button)
        button_panel.Add(self.__cancel_button, 0, wx.RIGHT, 5)

        self.__close_button = wx.Button(panel, label="Close")
        self.Bind(wx.EVT_BUTTON, lambda event: self.do_ok(), self.__close_button)
        button_panel.Add(self.__close_button, 0, 0, 0)
        self.__close_button.Hide()

        main_layout.Add(button_panel, 0, wx.ALIGN_CENTER, 0)
        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        panel.SetSizerAndFit(general_layout)
        self.Centre()
        self.Fit()
        self.__panel = panel

    def do_import(self):
        self.__import_button.Hide()
        self.__cancel_button.Hide()
        for editor in self.__valid_file_editors:
            filelist = [editor.get_fullname()]
            for tail in editor.get_tail_files():
                filelist.append(tail)
            try:
                if self.__do_links:
                    for file in filelist:
                        own_file = os.path.split(file)[1]
                        destination_file = os.path.join(self.__destination, own_file)
                        if os.path.isfile(destination_file):
                            raise RuntimeError("The same file is in the destination folder")
                        os.symlink(file, destination_file)
                    editor.set_processed()
                    self.Refresh()
                    self.Update()
                else:
                    idx = 0
                    for file in filelist:
                        editor.set_in_process(idx, len(filelist))
                        self.Refresh()
                        self.Update()
                        print("SRC {0} DST {1}".format(file, self.__destination))
                        own_file = os.path.split(file)[1]
                        destination_file = os.path.join(self.__destination, own_file)
                        if os.path.isfile(destination_file):
                            raise RuntimeError("The same file is in the destination folder")
                        shutil.copyfile(file, destination_file)
                        idx += 1
                    editor.set_processed()
                    self.Refresh()
                    self.Update()
            except Exception as err:
                editor.set_error(err)
                self.Refresh()
                self.Update()
                print(err)
        self.__close_button.Show()
        self.__panel.Layout()
        self.Layout()

    def do_cancel(self):
        self.EndModal(wx.ID_CANCEL)

    def do_ok(self):
        self.EndModal(wx.ID_OK)


class FileEditor:

    _layout = None
    _filetype = None
    _status_bar = None

    def __init__(self, scroll_panel, filename):
        self._layout = wx.BoxSizer(wx.HORIZONTAL)
        if self._filetype == "compressed" or self._filetype == "stream":
            filename += ", ..."

        filename_box = wx.StaticText(scroll_panel, label=filename,
                                     style=wx.ALIGN_LEFT | wx.ST_NO_AUTORESIZE | wx.ST_ELLIPSIZE_END)
        size = filename_box.GetSize()
        filename_box.SetSizeHints(150, size[1])
        self._layout.Add(filename_box, 0, wx.RIGHT, 5)

        filemode_box = wx.StaticText(scroll_panel, label=self._filetype,
                                     style=wx.ALIGN_LEFT | wx.ST_NO_AUTORESIZE | wx.ST_ELLIPSIZE_END)
        size = filemode_box.GetSize()
        filemode_box.SetSizeHints(100, size[1])
        self._layout.Add(filemode_box, 0, wx.RIGHT, 5)

        self._status_bar = wx.StaticText(scroll_panel, label="Reading file header...")
        font = self._status_bar.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self._status_bar.SetFont(font)
        self._layout.Add(self._status_bar, 1, wx.EXPAND, 0)

    def get_layout(self):
        return self._layout


class ValidFileEditor(FileEditor):

    __tail_files = None
    __fullname = None

    def __init__(self, scroll_panel, filename, filemode, tail_files):
        self._filetype = filemode
        super().__init__(scroll_panel, os.path.split(filename)[1])
        self.__fullname = filename
        self.__tail_files = tail_files

    def set_error(self, err):
        self._status_bar.SetLabel("Data import failed: {0}".format(err))
        self._status_bar.SetForegroundColour("red")

    def set_processed(self):
        self._status_bar.SetLabel("Done")
        self._status_bar.SetForegroundColour("green")

    def set_in_process(self, steps_completed, steps_total):
        self._status_bar.SetLabel("Copying...({0:.0%})".format(steps_completed/steps_total))
        self._status_bar.SetForegroundColour("orange")

    def set_ready(self):
        self._status_bar.SetLabel("Ready")
        self._status_bar.SetForegroundColour("orange")

    def get_fullname(self):
        return self.__fullname

    def get_tail_files(self):
        return self.__tail_files


class InvalidFileEditor(FileEditor):

    def __init__(self, scroll_panel, filename):
        self._filetype = ""
        super().__init__(scroll_panel, os.path.split(filename)[1])
        self._status_bar.SetLabel("Invalid")
        self._status_bar.SetForegroundColour("red")
