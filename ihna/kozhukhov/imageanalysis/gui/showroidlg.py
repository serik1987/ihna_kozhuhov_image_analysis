# -*- coding: utf-8

import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches


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
            vessel_axes.imshow(vessel_map, cmap="gray")
            self.__draw_roi(vessel_axes, roi)
        vessel_axes.set_aspect('equal')
        vessel_axes.set_title("Vessel map")
        vessel_axes.set_ylabel("Y")

        amplitude_axes = fig.add_subplot(2, 2, 3)
        if amplitude_map is not None:
            amplitude_map.imshow(amplitude_map)
            self.__draw_roi(vessel_axes, roi)
        amplitude_axes.set_aspect('equal')
        amplitude_axes.set_title("Amplitude map")
        amplitude_axes.set_ylabel("Y")
        amplitude_axes.set_xlabel("X")

        phase_axes = fig.add_subplot(2, 2, 4)
        if phase_map is not None:
            phase_map.imshow(phase_map)
            self.__draw_roi(vessel_axes, roi)
        phase_axes.set_aspect('equal')
        phase_axes.set_title("Phase map")
        phase_axes.set_xlabel("X")

        canvas = FigureCanvas(main_panel, -1, fig)
        sizer.Add(canvas, 1, wx.LEFT | wx.TOP | wx.EXPAND)
        main_panel.SetSizer(sizer)
        self.Centre()

    def __draw_roi(self, ax, roi, color="red"):
        if roi.get_type() == "simple":
            print("Simple ROI detected:", roi.get_name())
            xy = (roi.get_left(), roi.get_top())
            width = roi.get_right() - roi.get_left()
            height = roi.get_bottom() - roi.get_top()
            roi_image = patches.Rectangle(xy, width, height,
                                          fill=False, color=color)
            ax.add_patch(roi_image)
        if roi.get_type() == "complex":
            for subroi in roi.get_subroi_list():
                self.__draw_roi(ax, subroi, color)
