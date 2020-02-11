# -*- coding: utf-8

import os
import numpy as np
import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure


class FinalTracesDlg(wx.Dialog):
    """
    This dialog is used for the results that are finally processed
    """

    __traces = None

    __save_npz_box = None
    __add_to_manifest_box = None
    __save_mat_box = None
    __save_png_box = None

    __fig = None
    __canvas = None

    __ref_ax = None
    __pix_ax = None
    __ref_psd_ax = None
    __pix_psd_ax = None

    __prefix_box = None
    __postfix_box = None

    def __init__(self, parent, traces):
        super().__init__(parent, title="Trace analysis result", size=(800, 600))
        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)

        draw_panel = wx.Panel(main_panel, size=(800, 500))
        draw_panel.SetBackgroundColour("green")
        draw_sizer = wx.BoxSizer(wx.VERTICAL)
        self.__fig = Figure(tight_layout=True)
        caption_font = {"fontsize": 10}
        times = traces.get_times()
        freqs = traces.get_frequencies()
        fidx = freqs < 10
        freqs = freqs[fidx]

        self.__ref_ax = self.__fig.add_subplot(2, 2, 1)
        self.__ref_ax.plot(times, traces.get_reference())
        self.__ref_ax.set_xlim((times[0], times[-1]))
        self.__ref_ax.get_xaxis().set_ticks([])
        self.__ref_ax.set_ylabel("Reference signal", fontdict=caption_font)

        self.__ref_psd_ax = self.__fig.add_subplot(2, 2, 2)
        self.__ref_psd_ax.plot(freqs, traces.get_reference_psd()[fidx])
        self.__ref_psd_ax.set_xscale("log")
        self.__ref_psd_ax.get_xaxis().set_ticks([])
        # self.__ref_psd_ax.get_yaxis().set_ticks([])

        self.__pix_ax = self.__fig.add_subplot(2, 2, 3)
        self.__pix_ax.plot(times, traces.get_avg_trace())
        self.__pix_ax.set_xlim((times[0], times[-1]))
        self.__pix_ax.set_xlabel("Time, ms", fontdict=caption_font)
        self.__pix_ax.set_ylabel("Intrinsic signal")

        self.__pix_psd_ax = self.__fig.add_subplot(2, 2, 4)
        self.__pix_psd_ax.plot(freqs, traces.get_avg_psd()[fidx])
        self.__pix_psd_ax.set_xscale("log")
        # self.__pix_psd_ax.get_yaxis().set_ticks([])
        self.__pix_psd_ax.set_xlabel("Frequency, Hz", fontdict=caption_font)

        for ax in [self.__ref_ax, self.__ref_psd_ax, self.__pix_ax, self.__pix_psd_ax]:
            ax.tick_params(labelsize=10)

        self.__canvas = FigureCanvas(draw_panel, -1, self.__fig)
        draw_sizer.Add(self.__canvas, 1, wx.EXPAND)
        draw_panel.SetSizer(draw_sizer)
        main_layout.Add(draw_panel, 0, wx.BOTTOM, 10)

        middle_panel = wx.BoxSizer(wx.HORIZONTAL)
        save_panel = wx.BoxSizer(wx.VERTICAL)

        self.__save_npz_box = wx.CheckBox(main_panel, label="Save to NPZ")
        self.__save_npz_box.SetValue(True)
        self.__save_npz_box.Bind(wx.EVT_CHECKBOX, lambda event: self.define_manifest_enability())
        save_panel.Add(self.__save_npz_box, 0, wx.BOTTOM, 5)

        add_to_manifest_wrapped = wx.BoxSizer(wx.HORIZONTAL)
        self.__add_to_manifest_box = wx.CheckBox(main_panel, label="Add to manifest")
        self.__add_to_manifest_box.SetValue(True)
        add_to_manifest_wrapped.Add(self.__add_to_manifest_box, 0, wx.LEFT, 20)
        save_panel.Add(add_to_manifest_wrapped, 0, wx.BOTTOM, 5)

        self.__save_mat_box = wx.CheckBox(main_panel, label="Save to MAT")
        save_panel.Add(self.__save_mat_box, 0, wx.BOTTOM, 5)

        self.__save_png_box = wx.CheckBox(main_panel, label="Save to PNG")
        save_panel.Add(self.__save_png_box)

        middle_panel.Add(save_panel, 0, wx.RIGHT, 5)
        names_panel = wx.FlexGridSizer(2, 5, 5)

        prefix_caption = wx.StaticText(main_panel, label="Name prefix")
        names_panel.Add(prefix_caption, 0, wx.ALIGN_CENTER)

        self.__prefix_box = wx.TextCtrl(main_panel)
        names_panel.Add(self.__prefix_box)

        postfix_caption = wx.StaticText(main_panel, label="Name postfix")
        names_panel.Add(postfix_caption, 0, wx.ALIGN_CENTER)

        self.__postfix_box = wx.TextCtrl(main_panel)
        names_panel.Add(self.__postfix_box)

        middle_panel.Add(names_panel)
        main_layout.Add(middle_panel, 0, wx.BOTTOM, 10)

        buttons_panel = wx.BoxSizer(wx.HORIZONTAL)

        button_ok = wx.Button(main_panel, label="Save and close")
        self.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_OK), button_ok)
        buttons_panel.Add(button_ok)

        main_layout.Add(buttons_panel, 0, wx.ALIGN_CENTER)

        general_layout.Add(main_layout, 1, wx.EXPAND | wx.ALL, 10)
        main_panel.SetSizerAndFit(general_layout)
        self.Centre()
        self.Fit()

        self.__traces = traces

    def close(self):
        """
        Cleans memory from the data associated with a window
        """
        pass

    def define_manifest_enability(self):
        self.__add_to_manifest_box.Enable(self.__save_npz_box.IsChecked())

    def save_files(self, folder):
        """
        Saves the data to all files selected by the user. The method shall be called after closing the modal dialog.
        Does not add to the manifest.

        Arguments:
            folder - a folder where all files shall be places
        Return:
            If user want these traces to be added to the manifest, the function returns name of the NPZ-file that
            shall be included into the manifest

            If user doesn't want traces to be added to the manifest, the function returns None
        """
        npz_file = None
        self.__traces.set_prefix_name(self.__prefix_box.GetValue())
        self.__traces.set_postfix_name(self.__postfix_box.GetValue())
        if self.__save_npz_box.IsChecked():
            npz_file = self.__traces.save_npz(folder)
            if not self.__add_to_manifest_box.IsChecked():
                npz_file = None
        if self.__save_mat_box.IsChecked():
            self.__traces.save_mat(folder)
        if self.__save_png_box.IsChecked():
            png_file = self.__traces.get_fullname() + "_new.png"
            png_fullfile = os.path.join(folder, png_file)
            self.__fig.savefig(png_fullfile)

        return npz_file
