# -*- coding: utf-8

import wx
import numpy as np
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import scipy.io
import ihna.kozhukhov.imageanalysis.sourcefiles as sfiles


class FrameViewer(wx.Dialog):
    """
    This window represents the frame viewer
    """

    __train = None
    __frame_number = -1
    __current_frame = None
    __frame_body = None
    __total_frames = -1

    __btn_first = None
    __btn_previous = None
    __btn_go_to = None
    __btn_next = None
    __btn_last = None
    __btn_save_png = None
    __btn_save_npy = None
    __btn_save_mat = None
    __btn_define_roi = None

    __txt_number = None
    __txt_sequential_number = None
    __txt_arrival_time = None
    __txt_delay = None
    __txt_synch = None
    __constructed = False

    __canvas = None
    __fig = None
    __ax = None

    def __init__(self, parent, train):
        self.__train = train
        self.__frame_number = 0
        self.__total_frames = self.__train.total_frames
        if isinstance(self.__train, sfiles.CompressedFileTrain):
            self.__total_frames = 1
        self.__constructed = False
        super().__init__(parent, title="Frame viewer: " + self.__train.filename, size=(800, 600))
        main_panel = wx.Panel(self)

        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)

        top_layout = wx.BoxSizer(wx.HORIZONTAL)

        left_layout = self.__create_left_layout(main_panel)
        top_layout.Add(left_layout, 5, wx.RIGHT | wx.EXPAND, 10)

        right_layout = self.__create_right_layout(main_panel)
        top_layout.Add(right_layout, 3, wx.EXPAND)
        main_layout.Add(top_layout, 1, wx.BOTTOM | wx.EXPAND, 50)

        bottom_layout = self.__create_bottom_layout(main_panel)
        main_layout.Add(bottom_layout, 0, wx.BOTTOM | wx.EXPAND, 50)

        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizer(general_layout)
        self.Centre()

        self.load_frame()

    def __create_left_layout(self, parent):
        left_layout = wx.Panel(parent, size=(400, 500))
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.__fig = Figure()
        self.__canvas = FigureCanvas(left_layout, -1, self.__fig)
        self.__ax = self.__fig.add_subplot(111)
        self.__ax.set_aspect('equal')
        self.__ax.set_title(self.__train.filename)
        self.__ax.plot([1, 2, 3], [1, 2, 3], 'b-')
        self.__ax.plot([1, 2, 3], [3, 2, 1], 'b-')

        sizer.Add(self.__canvas, 1, wx.LEFT | wx.TOP | wx.EXPAND)
        left_layout.SetSizer(sizer)
        return left_layout

    def __create_right_layout(self, parent):
        right_layout = wx.FlexGridSizer(2, 5, 5)

        label = wx.StaticText(parent, label="Frame number (after binning)")
        right_layout.Add(label)

        self.__txt_number = wx.StaticText(parent, label="undefined")
        right_layout.Add(self.__txt_number)

        label = wx.StaticText(parent, label="Frame number (before binning)")
        right_layout.Add(label)

        self.__txt_sequential_number = wx.StaticText(parent, label="undefined")
        right_layout.Add(self.__txt_sequential_number)

        label = wx.StaticText(parent, label="Frame arrival time")
        right_layout.Add(label)

        self.__txt_arrival_time = wx.StaticText(parent, label="undefined")
        right_layout.Add(self.__txt_arrival_time)

        label = wx.StaticText(parent, label="Delay")
        right_layout.Add(label)

        self.__txt_delay = wx.StaticText(parent, label="undefined")
        right_layout.Add(self.__txt_delay)

        if self.__train.experiment_mode == "continuous":
            self.__txt_synch = []
            for chan in range(self.__train.synchronization_channel_number):
                label = wx.StaticText(parent, label="Synchronization value (chan = {0})".format(chan))
                right_layout.Add(label)

                synch = wx.StaticText(parent, label="undefined")
                right_layout.Add(synch)
                self.__txt_synch.append(synch)

        return right_layout

    def __create_bottom_layout(self, parent):
        bottom_layout = wx.BoxSizer(wx.HORIZONTAL)

        self.__btn_first = wx.Button(parent, label="<<", size=(30, 20))
        self.Bind(wx.EVT_BUTTON, lambda event: self.first(), self.__btn_first)
        bottom_layout.Add(self.__btn_first, 0, wx.RIGHT, 5)

        self.__btn_previous = wx.Button(parent, label="<", size=(30, 20))
        self.Bind(wx.EVT_BUTTON, lambda event: self.previous(), self.__btn_previous)
        bottom_layout.Add(self.__btn_previous, 0, wx.RIGHT, 5)

        self.__btn_go_to = wx.Button(parent, label="Frame # 100000", size=(200, 20))
        self.Bind(wx.EVT_BUTTON, lambda event: self.go_to(), self.__btn_go_to)
        bottom_layout.Add(self.__btn_go_to, 0, wx.RIGHT, 5)

        self.__btn_next = wx.Button(parent, label=">", size=(30, 20))
        self.Bind(wx.EVT_BUTTON, lambda event: self.next(), self.__btn_next)
        bottom_layout.Add(self.__btn_next, 0, wx.RIGHT, 5)

        self.__btn_last = wx.Button(parent, label=">>", size=(30, 20))
        self.Bind(wx.EVT_BUTTON, lambda event: self.last(), self.__btn_last)
        bottom_layout.Add(self.__btn_last, 0, wx.RIGHT, 100)

        self.__btn_save_png = wx.Button(parent, label="PNG", size=(50, 20))
        self.Bind(wx.EVT_BUTTON, lambda event: self.save_png(), self.__btn_save_png)
        bottom_layout.Add(self.__btn_save_png, 0, wx.RIGHT, 5)

        self.__btn_save_npy = wx.Button(parent, label="NPY", size=(50, 20))
        self.Bind(wx.EVT_BUTTON, lambda event: self.save_npy(), self.__btn_save_npy)
        bottom_layout.Add(self.__btn_save_npy, 0, wx.RIGHT, 5)

        self.__btn_save_mat = wx.Button(parent, label="MAT", size=(50, 20))
        self.Bind(wx.EVT_BUTTON, lambda event: self.save_mat(), self.__btn_save_mat)
        bottom_layout.Add(self.__btn_save_mat, 0, wx.RIGHT, 5)

        self.__btn_define_roi = wx.Button(parent, label="ROI", size=(50, 20))
        self.Bind(wx.EVT_BUTTON, lambda event: self.define_roi(), self.__btn_define_roi)
        bottom_layout.Add(self.__btn_define_roi, 0, wx.RIGHT, 5)

        return bottom_layout

    def load_frame(self):
        self.__current_frame = self.__train[self.__frame_number]
        self.__frame_body = self.__current_frame.body
        self.__ax.cla()
        self.__ax.imshow(self.__frame_body, cmap='gray')
        self.__ax.set_title("{0}: Frame # {1}".format(self.__train.filename, self.__frame_number))
        self.__txt_number.SetLabel(str(self.__current_frame.number))
        self.__txt_sequential_number.SetLabel(str(self.__current_frame.sequential_number))
        self.__txt_arrival_time.SetLabel("{0:.0f} ms".format(self.__current_frame.arrival_time))
        self.__txt_delay.SetLabel("{0:.0f} ms".format(self.__current_frame.delay))
        if self.__train.experiment_mode == "continuous":
            synch = self.__current_frame.synch
            for i in range(self.__train.synchronization_channel_number):
                self.__txt_synch[i].SetLabel(str(synch[i]))
        prev_enability = self.__frame_number != 0
        next_enability = self.__frame_number != self.__total_frames-1
        goto_enability = not isinstance(self.__train, sfiles.CompressedFileTrain)
        self.__btn_first.Enable(prev_enability)
        self.__btn_previous.Enable(prev_enability)
        self.__btn_go_to.Enable(goto_enability)
        self.__btn_go_to.SetLabel("Frame # {0}".format(self.__frame_number))
        self.__btn_next.Enable(next_enability)
        self.__btn_last.Enable(next_enability)
        self.__canvas.draw()

    def close(self):
        self.__current_frame = None
        self.__frame_body = None

    def first(self):
        old_number = self.__frame_number
        try:
            self.__frame_number = 0
            self.load_frame()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Frame Viewer", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()
            self.__frame_number = old_number

    def previous(self):
        old_number = self.__frame_number
        try:
            self.__frame_number -= 1
            self.load_frame()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Frame Viewer", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()
            self.__frame_number = old_number

    def go_to(self):
        old_number = self.__frame_number
        try:
            dlg = wx.TextEntryDialog(self, "Number of the frame to show", "Frame Viewer")
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            n = int(dlg.GetValue())
            if n < 0 or n >= self.__total_frames:
                raise ValueError("Frame index is out of range")
            self.__frame_number = n
            self.load_frame()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Frame Viewer", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()
            self.__frame_number = old_number

    def next(self):
        old_number = self.__frame_number
        try:
            self.__frame_number += 1
            self.load_frame()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Frame Viewer", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()
            self.__frame_number = old_number

    def last(self):
        old_number = self.__frame_number
        try:
            self.__frame_number = self.__total_frames - 1
            self.load_frame()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Frame Viewer", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()
            self.__frame_number = old_number

    def save_png(self):
        try:
            dlg = wx.FileDialog(self, "Save as PNG", style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT, wildcard=".png")
            dlg.ShowModal()
            filename = dlg.GetPath()
            if not filename.endswith(".png"):
                filename += ".png"
            self.__fig.savefig(filename)
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Frame Viewer", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def save_npy(self):
        try:
            dlg = wx.FileDialog(self, "Save as NPY", style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT, wildcard=".npy")
            dlg.ShowModal()
            filename = dlg.GetPath()
            if not filename.endswith(".npy"):
                filename += ".npy"
            np.save(filename, self.__frame_body)
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Frame Viewer", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def save_mat(self):
        try:
            dlg = wx.FileDialog(self, "Save as MAT", style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT, wildcard=".mat")
            dlg.ShowModal()
            filename = dlg.GetPath()
            if not filename.endswith(".mat"):
                filename += ".mat"
            scipy.io.savemat(filename, {"frame_data": self.__frame_body})
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Frame Viewer", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def define_roi(self):
        print("TO-DO: define ROI")
