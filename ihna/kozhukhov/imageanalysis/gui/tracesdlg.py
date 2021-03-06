# -*- coding: utf-8

import wx
from scipy import diff
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure


class TracesDlg(wx.Dialog):
    """
    Represents a window that shows traces with synchronization and without synchronization
    """

    __isoline_remove_checkbox = None
    __average_psd_box = None
    __average_signal_box = None
    __average_option_box = None
    __median_option_box = None

    __time_arrival_axes = None
    __reference_signal_axes = None
    __synchronization_axes = None

    __traces_before_correction_axes = None
    __isolines_axes = None
    __traces_after_correction_axes = None
    __traces_averaged_axes = None

    __psd_before_correction_axes = None
    __psd_isolines_axes = None
    __psd_after_correction_axes = None
    __psd_averaged_axes = None
    __psd_reference_axes = None

    __fig = None
    __fig_canvas = None

    def __init__(self, parent, processor):
        super().__init__(parent, title="Trace analysis", size=(800, 600))
        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)

        draw_panel = wx.Panel(main_panel, size=(900, 600))
        draw_panel.SetBackgroundColour("green")
        draw_sizer = wx.BoxSizer(wx.VERTICAL)
        fig = Figure(tight_layout=True)
        caption_font = {"fontsize": 10}
        fr = processor.get_frame_vector()
        diff_fr = fr[:-1]

        self.__time_arrival_axes = fig.add_subplot(5, 3, 1)
        delta = diff(processor.get_time_arrivals())
        self.__time_arrival_axes.plot(diff_fr, delta)
        m = delta.mean()
        s = delta.std()
        self.__time_arrival_axes.get_xaxis().set_ticks([])
        self.__time_arrival_axes.set_xlim(processor.get_frame_lim())
        self.__time_arrival_axes.set_ylim((m - 3 * s, m + 3 * s))
        self.__time_arrival_axes.get_yaxis().set_ticks([m - 2*s, m, m + 2*s])
        self.__time_arrival_axes.set_title("Arrival time difference, ms", fontdict=caption_font)
        self.__time_arrival_axes.tick_params(labelsize=10)

        self.__traces_before_correction_axes = fig.add_subplot(5, 3, 2)
        self.__traces_before_correction_axes.plot(processor.get_frame_vector(),
                                                  processor.get_data_not_removed()[:, 0:7])
        self.__traces_before_correction_axes.set_xlim(processor.get_frame_lim())
        self.__traces_before_correction_axes.get_xaxis().set_ticks([])
        self.__traces_before_correction_axes.set_title("Traces, before isoline remove", fontdict=caption_font)
        self.__traces_before_correction_axes.tick_params(labelsize=10)

        self.__psd_before_correction_axes = fig.add_subplot(5, 3, 3)
        self.__psd_before_correction_axes.plot(processor.get_psd_not_removed()[:, 0:7])
        self.__psd_before_correction_axes.get_xaxis().set_ticks([])
        self.__psd_before_correction_axes.set_title("Traces PSD, before isoline remove", fontdict=caption_font)
        self.__psd_before_correction_axes.tick_params(labelsize=10)

        self.__reference_signal_axes = fig.add_subplot(5, 3, 4)
        self.__reference_signal_axes.plot(processor.get_frame_vector(), processor.get_reference_signal())
        self.__reference_signal_axes.get_xaxis().set_ticks([])
        self.__reference_signal_axes.set_xlim(processor.get_frame_lim())
        self.__reference_signal_axes.set_title("Reference signal", fontdict=caption_font)
        self.__reference_signal_axes.tick_params(labelsize=10)

        self.__isolines_axes = fig.add_subplot(5, 3, 5)
        self.__isolines_axes.plot(processor.get_frame_vector(), processor.get_isolines()[:, 0:7])
        self.__isolines_axes.set_xlim(processor.get_frame_lim())
        self.__isolines_axes.get_xaxis().set_ticks([])
        self.__isolines_axes.set_title("Isolines", fontdict=caption_font)
        self.__isolines_axes.tick_params(labelsize=10)

        self.__psd_isolines_axes = fig.add_subplot(5, 3, 6)
        self.__psd_isolines_axes.plot(processor.get_isoline_psd()[:, 0:7])
        self.__psd_isolines_axes.get_xaxis().set_ticks([])
        self.__psd_isolines_axes.set_title("Isoline PSD", fontdict=caption_font)
        self.__psd_isolines_axes.tick_params(labelsize=10)

        self.__synchronization_axes = fig.add_subplot(5, 3, 7)
        chan_list = []
        for chan in range(processor.get_synch_channel_number()):
            chan_data = processor.get_synch_channel(chan)
            self.__synchronization_axes.plot(fr, chan_data)
            chan_list.append("{0}".format(chan))
        self.__synchronization_axes.set_xlabel("Timestamp", fontdict=caption_font)
        self.__synchronization_axes.set_xlim(processor.get_frame_lim())
        self.__synchronization_axes.set_title("Synchronization channels", fontdict=caption_font)
        self.__synchronization_axes.legend(chan_list, fontsize="small", loc="upper right")
        self.__synchronization_axes.tick_params(labelsize=10)

        self.__traces_after_correction_axes = fig.add_subplot(5, 3, 8)
        self.__traces_after_correction_axes.plot(processor.get_frame_vector(), processor.get_data()[:, 0:7])
        self.__traces_after_correction_axes.set_xlim(processor.get_frame_lim())
        self.__traces_after_correction_axes.get_xaxis().set_ticks([])
        self.__traces_after_correction_axes.set_title("Traces, after isoline remove", fontdict=caption_font)
        self.__traces_after_correction_axes.tick_params(labelsize=10)

        self.__psd_after_correction_axes = fig.add_subplot(5, 3, 9)
        self.__psd_after_correction_axes.plot(processor.get_psd()[:, 0:7])
        self.__psd_after_correction_axes.get_xaxis().set_ticks([])
        self.__psd_after_correction_axes.set_title("Traces PSD, after isoline remove", fontdict=caption_font)
        self.__psd_after_correction_axes.tick_params(labelsize=10)

        self.__traces_averaged_axes = fig.add_subplot(5, 3, 11)
        self.__traces_averaged_axes.plot(processor.get_frame_vector(), processor.get_average_signal(), 'b-')
        self.__traces_averaged_axes.plot(processor.get_frame_vector(), processor.get_median_signal(), 'r--')
        self.__traces_averaged_axes.set_xlim(processor.get_frame_lim())
        self.__traces_averaged_axes.set_xlabel("# frame", fontdict=caption_font)
        self.__traces_averaged_axes.set_title("Traces, averaged", fontdict=caption_font)
        self.__traces_averaged_axes.tick_params(labelsize=10)

        self.__psd_averaged_axes = fig.add_subplot(5, 3, 12)
        self.__psd_averaged_axes.plot(processor.get_average_signal_spectrum(), 'b-')
        self.__psd_averaged_axes.plot(processor.get_median_signal_spectrum(), 'b--')
        self.__psd_averaged_axes.plot(processor.get_average_spectrum(), 'g-')
        self.__psd_averaged_axes.plot(processor.get_median_spectrum(), 'g--')
        self.__psd_averaged_axes.get_xaxis().set_ticks([])
        self.__psd_averaged_axes.set_title("Traces PSD, averaged", fontdict=caption_font)
        self.__psd_averaged_axes.tick_params(labelsize=10)

        self.__psd_reference_axes = fig.add_subplot(5, 3, 15)
        self.__psd_reference_axes.plot(processor.get_reference_spectrum())
        self.__psd_reference_axes.set_xlabel("Frequency, d.u.", fontdict=caption_font)
        self.__psd_reference_axes.set_title("PSD of the reference signal", fontdict=caption_font)
        self.__psd_reference_axes.tick_params(labelsize=10)

        canvas = FigureCanvas(draw_panel, -1, fig)
        draw_sizer.Add(canvas, 1, wx.EXPAND)
        draw_panel.SetSizer(draw_sizer)
        main_layout.Add(draw_panel, 0, wx.BOTTOM, 10)

        self.__fig = fig
        self.__fig_canvas = canvas

        options_panel = wx.BoxSizer(wx.HORIZONTAL)
        average_strategy_box = wx.BoxSizer(wx.VERTICAL)

        self.__average_signal_box = wx.RadioButton(main_panel, label="Plot PSD of averaged (blue lines)",
                                                   style=wx.RB_GROUP)
        average_strategy_box.Add(self.__average_signal_box, 0, wx.BOTTOM, 5)

        self.__average_psd_box = wx.RadioButton(main_panel, label="Plot Average PSDs (green lines)")
        average_strategy_box.Add(self.__average_psd_box, 0, wx.BOTTOM, 5)

        options_panel.Add(average_strategy_box, 0, wx.RIGHT | wx.EXPAND, 5)
        average_method_box = wx.BoxSizer(wx.VERTICAL)

        self.__average_option_box = wx.RadioButton(main_panel, label="Plot mean (solid lines)",
                                                   style=wx.RB_GROUP)
        average_method_box.Add(self.__average_option_box, 0, wx.BOTTOM, 5)

        self.__median_option_box = wx.RadioButton(main_panel, label="Plot median (dashed lines)")
        average_method_box.Add(self.__median_option_box)

        options_panel.Add(average_method_box, 0, wx.RIGHT, 5)

        wx.StaticText(main_panel, label=processor.get_annotation_text(), pos=(10, 550))

        main_layout.Add(options_panel, 0, wx.BOTTOM, 10)

        buttons_panel = wx.BoxSizer(wx.HORIZONTAL)

        button_ok = wx.Button(main_panel, label="Continue trace analysis")
        button_ok.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_OK))
        buttons_panel.Add(button_ok, wx.RIGHT, 5)

        button_cancel = wx.Button(main_panel, label="Cancel")
        button_cancel.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_CANCEL))
        buttons_panel.Add(button_cancel)

        main_layout.Add(buttons_panel, 0, wx.ALIGN_CENTER)

        general_layout.Add(main_layout, 0, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizerAndFit(general_layout)
        self.Centre()
        self.Fit()

    def set_average_method_and_strategy(self, processor):
        if self.__average_signal_box.GetValue():
            processor.set_average_strategy('average_than_psd')
        if self.__average_psd_box.GetValue():
            processor.set_average_strategy('psd_than_average')
        if self.__average_option_box.GetValue():
            processor.set_average_method('mean')
        if self.__median_option_box.GetValue():
            processor.set_average_method('median')

    def close(self):
        self.__fig_canvas = None
        self.__fig = None
