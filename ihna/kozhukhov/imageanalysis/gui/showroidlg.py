# -*- coding: utf-8

import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure


class ShowRoiDlg(wx.Dialog):
    """
    This is a simple box to show all ROIs

    Input arguments:
        parent - the parent window
        fullname - name of the case
        roi - ROI to show
        vessel_map - one of the frames or None if not available
        amplitude_map - the amplitude map or None if not available
        phase_map - the phase ap or None if not available
    """

    def __init__(self, parent, fullname, roi, vessel_map=None, amplitude_map=None, phase_map=None):
        super().__init__(parent, title="Maps after ROI application: " + fullname, size=(800, 800))
        main_panel = wx.Panel(self)
        main_panel.SetBackgroundColour("red")
        sizer = wx.BoxSizer(wx.VERTICAL)

        fig = Figure(tight_layout=True)

        vessel_axes = fig.add_subplot(2, 2, 1)
        if vessel_map is not None:
            vessel_map = roi.apply(vessel_map)
            vessel_axes.imshow(vessel_map, cmap="gray")
        vessel_axes.set_aspect('equal')
        vessel_axes.set_title("Vessel map")
        vessel_axes.set_ylabel("Y")

        amplitude_axes = fig.add_subplot(2, 2, 3)
        if amplitude_map is not None:
            amplitude_map = roi.apply(amplitude_map)
            amplitude_map.imshow(amplitude_map)
        amplitude_axes.set_aspect('equal')
        amplitude_axes.set_title("Amplitude map")
        amplitude_axes.set_ylabel("Y")
        amplitude_axes.set_xlabel("X")

        phase_axes = fig.add_subplot(2, 2, 4)
        if phase_map is not None:
            phase_map = roi.apply(phase_map)
            phase_map.imshow(phase_map)
        phase_axes.set_aspect('equal')
        phase_axes.set_title("Phase map")
        phase_axes.set_xlabel("X")

        canvas = FigureCanvas(main_panel, -1, fig)
        sizer.Add(canvas, 1, wx.LEFT | wx.TOP | wx.EXPAND)
        main_panel.SetSizer(sizer)
        self.Centre()
