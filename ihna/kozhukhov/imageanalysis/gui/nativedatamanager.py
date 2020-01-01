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
        left_panel = wx.Notebook(main_panel, style=wx.BK_DEFAULT)

        general_properties_page = self.__create_general_properties(left_panel)
        left_panel.AddPage(general_properties_page, "General Properties")

        for file in self.__train:
            break

        for chunk in file.isoi:
            if chunk['id'] != "SOFT":
                viewer = ChunkViewer.new_viewer(left_panel, chunk)
                left_panel.AddPage(viewer, viewer.get_title())

        return left_panel

    def __create_general_properties(self, parent):
        general_properties_page = wx.Panel(parent)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.FlexGridSizer(2, 5, 15)

        label = wx.StaticText(general_properties_page, label="Number of files in the train")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=str(self.__train.file_number))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Total number of frames")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=str(self.__train.total_frames))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Frame dimensions")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} x {1} pixels".format(self.__train.frame_shape[1],
                                                                                       self.__train.frame_shape[0]))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Total number of pixels in the frame")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=str(self.__train.frame_size))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Frame body size")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} bytes".format(self.__train.frame_image_size))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Frame header size")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} bytes".format(self.__train.frame_header_size))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Total frame size (header + body)")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} bytes".format(self.__train.total_frame_size))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="File header size")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} bytes".format(self.__train.file_header_size))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Experiment mode")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=self.__train.experiment_mode)
        main_layout.Add(label)

        for file in self.__train:
            break
        soft = file.soft

        label = wx.StaticText(general_properties_page, label="Date and time of the record")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=soft['date_time_recorded'])
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="User name")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=soft['user_name'])
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Subject ID")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=soft['subject_id'])
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Pixel size")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} bytes".format(soft['pixel_size']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="ROI position (before binning)")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="({0}, {1}) pixels".format(soft['roi_x_position'],
                                                                                        soft['roi_y_position']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="ROI position (adjusted, before binning)")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="({0}, {1}) pixels".format(soft['roi_x_position_adjusted'],
                                                                                        soft['roi_y_position_adjusted']
                                                                                        ))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="ROI size (before binning)")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} x {1} pixels".format(soft['roi_x_size'],
                                                                                       soft['roi_y_size']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="ROI number")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=str(soft['roi_number']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Number of bins for temporal binning")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=str(soft['temporal_binning']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Spatial binning")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} x {1} pixels".format(soft['spatial_binning_x'],
                                                                                       soft['spatial_binning_y']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Wavelength")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} nm".format(soft['wavelength']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Filter width")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} nm".format(soft['filter_width']))
        main_layout.Add(label)

        general_layout.Add(main_layout, 1, wx.EXPAND | wx.ALL, 10)
        general_properties_page.SetSizer(general_layout)
        general_properties_page.Layout()
        return general_properties_page

    def __create_right_panel(self, main_panel):
        right_panel = wx.Notebook(main_panel)
        right_panel.SetBackgroundColour("blue")

        return right_panel

    def close(self):
        """
        Closes the file opened by this dialog
        """
        self.__train.close()
