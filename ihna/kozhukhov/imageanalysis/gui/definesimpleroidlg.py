# -*- coding: utf-8

from PIL import Image
import wx
import numpy as np
from ihna.kozhukhov.imageanalysis.manifest import SimpleRoi


class DefineSimpleRoiDlg(wx.Dialog):
    """
    This dialog is to define the simple ROI
    """

    __vessel_map_widget = None
    __complex_map_widget = None
    __amplitude_map_widget = None
    __phase_map_widget = None

    __name_box = None

    __left_box = None
    __right_box = None
    __top_box = None
    __bottom_box = None
    __width_box = None
    __height_box = None
    __area_box = None

    __current_border = "left"
    __roi = None

    def __init__(self, parent, fullname, vessel_map=None, amplitude_map=None, phase_map=None, harmonic=1.0):
        """
        Creates new dialog

        Arguments:
            parent - the parent window
            fullname - full name of the case (doesn't affect on the program functionality)
            vessel_map - some frame from the native data or None if this is not available
            amplitude_map - the amplitude map or None if not available
            phase_map - the phase map or None if not available
            harmonic - the harmonic ratio

        After you ShowModal it:
            get_roi() - ROI that was defined by the user
        """
        self.__init_roi(vessel_map, amplitude_map, phase_map)
        super().__init__(parent, title="Define simple ROI: " + fullname, size=(830, 620))
        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.HORIZONTAL)

        maps = self.__create_maps(main_panel, vessel_map, amplitude_map, phase_map, harmonic)
        main_layout.Add(maps, 6, wx.RIGHT | wx.EXPAND, 10)

        controls = self.__create_controls(main_panel)
        main_layout.Add(controls, 2, wx.EXPAND)

        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizer(general_layout)
        self.Centre()

        self.reload_roi()

    def __create_maps(self, parent, vessel_map, amplitude_map, phase_map, harmonic):
        if vessel_map is not None:
            vessel_map = (vessel_map - vessel_map.min()) / (vessel_map.max() - vessel_map.min())

        if amplitude_map is not None:
            amplitude_map = (amplitude_map - amplitude_map.min()) / (amplitude_map.max() - amplitude_map.min())

        if phase_map is not None:
            phase_map = np.exp(1j * phase_map * harmonic)

        if amplitude_map is not None and phase_map is not None:
            complex_map = amplitude_map * phase_map
        else:
            complex_map = None

        maps = wx.GridSizer(2, 5, 5)

        self.__vessel_map_widget = DefineSimpleRoiPanel(self, parent, vessel_map, "gray", "red", self.__roi)
        maps.Add(self.__vessel_map_widget, 1, wx.EXPAND)

        self.__complex_map_widget = DefineSimpleRoiPanel(self, parent, complex_map, "hsv", "green", self.__roi)
        maps.Add(self.__complex_map_widget, 1, wx.EXPAND)

        self.__amplitude_map_widget = DefineSimpleRoiPanel(self, parent, amplitude_map, "gray", "blue", self.__roi)
        maps.Add(self.__amplitude_map_widget, 1, wx.EXPAND)

        self.__phase_map_widget = DefineSimpleRoiPanel(self, parent, phase_map, "hsv", "yellow", self.__roi)
        maps.Add(self.__phase_map_widget, 1, wx.EXPAND)

        return maps

    def __set_radio(self, value):
        self.__current_border = value

    def finalize_roi_definition(self):
        name = self.__name_box.GetValue()
        if name == "":
            dlg = wx.MessageDialog(self, "Please, specify the ROI name", "Define simple ROI",
                                   wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()
            return
        self.__roi.set_name(name)
        self.EndModal(wx.OK)

    def get_roi(self):
        return self.__roi

    def __create_controls(self, parent):
        controls = wx.BoxSizer(wx.VERTICAL)

        name_prompt = wx.StaticText(parent, label="ROI name (*)")
        controls.Add(name_prompt, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__name_box = wx.TextCtrl(parent)
        controls.Add(self.__name_box, 0, wx.EXPAND | wx.BOTTOM, 25)

        left_radio = wx.RadioButton(parent, label="Left border", style=wx.RB_GROUP)
        self.Bind(wx.EVT_RADIOBUTTON, lambda event: self.__set_radio("left"), left_radio)
        controls.Add(left_radio, 0, wx.BOTTOM | wx.EXPAND, 5)

        right_radio = wx.RadioButton(parent, label="Right border")
        self.Bind(wx.EVT_RADIOBUTTON, lambda event: self.__set_radio("right"), right_radio)
        controls.Add(right_radio, 0, wx.BOTTOM | wx.EXPAND, 5)

        top_radio = wx.RadioButton(parent, label="Top border")
        self.Bind(wx.EVT_RADIOBUTTON, lambda event: self.__set_radio("top"), top_radio)
        controls.Add(top_radio, 0, wx.BOTTOM | wx.EXPAND, 5)

        bottom_radio = wx.RadioButton(parent, label="Bottom border")
        self.Bind(wx.EVT_RADIOBUTTON, lambda event: self.__set_radio("bottom"), bottom_radio)
        controls.Add(bottom_radio, 0, wx.BOTTOM | wx.EXPAND, 25)

        subcontrols = wx.GridSizer(2, 5, 5)

        self.__left_box = wx.StaticText(parent, label="left = 1024")
        subcontrols.Add(self.__left_box, 0, wx.EXPAND)

        self.__right_box = wx.StaticText(parent, label="right = 1024")
        subcontrols.Add(self.__right_box, 0, wx.EXPAND)

        self.__bottom_box = wx.StaticText(parent, label="bottom = 1024")
        subcontrols.Add(self.__bottom_box, 0, wx.EXPAND)

        self.__top_box = wx.StaticText(parent, label="top = 1024")
        subcontrols.Add(self.__top_box, 0, wx.EXPAND)

        self.__width_box = wx.StaticText(parent, label="width = 1024")
        subcontrols.Add(self.__width_box, 0, wx.EXPAND)

        self.__height_box = wx.StaticText(parent, label="height = 1024")
        subcontrols.Add(self.__height_box, 0, wx.EXPAND)

        controls.Add(subcontrols, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__area_box = wx.StaticText(parent, label="area = {0}".format(1024**2), style=wx.ALIGN_CENTRE_HORIZONTAL)
        controls.Add(self.__area_box, 1, wx.BOTTOM | wx.EXPAND, 50)

        buttons = wx.BoxSizer(wx.HORIZONTAL)

        ok = wx.Button(parent, label="OK")
        self.Bind(wx.EVT_BUTTON, lambda event: self.finalize_roi_definition(), ok)
        buttons.Add(ok, 0, wx.RIGHT, 5)

        cancel = wx.Button(parent, label="Cancel")
        self.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_CANCEL), cancel)
        buttons.Add(cancel)

        controls.Add(buttons, 0, wx.ALIGN_CENTER)
        return controls

    def __init_roi(self, vessel_map, amplitude_map, phase_map):
        working_map = None
        if vessel_map is not None:
            working_map = vessel_map
        if amplitude_map is not None:
            working_map = amplitude_map
        if phase_map is not None:
            working_map = phase_map
        if working_map is None:
            raise ValueError("at least, one of vessel_map or amplitude_map or phase_map shall not be None")

        self.__roi = SimpleRoi()
        self.__roi.set_left(0)
        self.__roi.set_right(working_map.shape[1])
        self.__roi.set_top(0)
        self.__roi.set_bottom(working_map.shape[1])
        self.__roi.set_name("undefined")

    def define_roi_position(self, x, y):
        try:
            if self.__current_border == "left":
                self.__roi.set_left(x)
            if self.__current_border == "right":
                self.__roi.set_right(x)
            if self.__current_border == "top":
                self.__roi.set_top(y)
            if self.__current_border == "bottom":
                self.__roi.set_bottom(y)
        except Exception as err:
            print(err)
        self.Refresh()
        self.reload_roi()

    def reload_roi(self):
        try:
            self.__left_box.SetLabel("left = {0}".format(self.__roi.get_left()))
            self.__right_box.SetLabel("Right = {0}".format(self.__roi.get_right()))
            self.__top_box.SetLabel("Top = {0}".format(self.__roi.get_top()))
            self.__bottom_box.SetLabel("Bottom = {0}".format(self.__roi.get_bottom()))
            self.__width_box.SetLabel("Width = {0}".format(self.__roi.get_width()))
            self.__height_box.SetLabel("Height = {0}".format(self.__roi.get_height()))
            self.__area_box.SetLabel("Area = {0}".format(self.__roi.get_area()))
        except Exception as err:
            print(err)


class DefineSimpleRoiPanel(wx.Panel):
    """
    Represents a panel that plots single maps to draw the ROI
    """

    __map = None
    __roi = None
    __colormap = None
    __dlg = None
    __old_height = None
    __old_width = None
    __bitmap_height = None
    __bitmap_width = None
    __ratio = None
    __bitmap = None

    def __init__(self, dlg, parent, data_map, colormap, placeholder, roi):
        """
        Arguments:
            dlg - the dialog itself
            parent - the parent widget
            data_map - numpy array of complex values, which amplitudes shall be in range [0, 1]
            colormap - matplotlib's colormap to be drawn, 'gray' and 'hsv parameters only are supported
            placeholder - what shall be placed if the map is None
        """
        super().__init__(parent)
        self.SetBackgroundColour(placeholder)
        self.__dlg = dlg
        self.__roi = roi

        if data_map is not None:
            if colormap == "gray" or colormap == "hsv":
                self.__colormap = colormap
            else:
                raise ValueError("The colormap may be either 'gray' or 'hsv'")
            self.__map = data_map
            self.Bind(wx.EVT_PAINT, self.paint)
            self.Bind(wx.EVT_LEFT_DOWN, self.__left_mouse_click)

        else:
            self.__map = None
            self.__colormap = None
            self.__dlg = None

    def paint(self, e):
        print("PY Repaint")

        dc = wx.PaintDC(self)
        brush = wx.Brush("white")
        dc.SetBackground(brush)
        dc.Clear()

        self.__paint_bitmap(dc)
        self.__paint_roi(dc)

    def __paint_bitmap(self, dc):
        initial_height, initial_width = self.__map.shape
        final_height = self.GetSize().GetHeight()
        final_width = self.GetSize().GetWidth()
        if final_height != self.__old_height or final_width != self.__old_width:
            self.__old_height = final_height
            self.__old_width = final_width
            height_ratio = final_height / initial_height
            width_ratio = final_width / initial_width
            ratio = min(height_ratio, width_ratio)
            bitmap_height = int(initial_height * ratio)
            bitmap_width = int(initial_width * ratio)
            pixel_map = None

            if self.__colormap == "gray":
                raw_pixel_map = np.array(self.__map * 255, dtype=np.uint8)
                pixel_map = np.zeros((initial_height, initial_width, 3), dtype=np.uint8)
                for d in range(3):
                    pixel_map[:, :, d] = raw_pixel_map
                pixel_map = Image.fromarray(pixel_map, 'RGB')

            if self.__colormap == "hsv":
                raw_hue_map = np.angle(self.__map) * 256 / (2 * np.pi)
                hue_map = np.array(raw_hue_map, dtype=np.uint8)
                raw_saturation_map = np.abs(self.__map) * 255
                saturation_map = np.array(raw_saturation_map, dtype=np.uint8)
                value_map = saturation_map
                raw_pixel_map = np.zeros((initial_height, initial_width, 3), dtype=np.uint8)
                raw_pixel_map[:, :, 0] = hue_map
                raw_pixel_map[:, :, 1] = saturation_map
                raw_pixel_map[:, :, 2] = value_map
                pixel_map = Image.fromarray(raw_pixel_map, "HSV")
                pixel_map = pixel_map.convert(mode="RGB")

            if pixel_map is not None:
                if ratio <= 1.0:
                    pixel_map = pixel_map.resize((bitmap_width, bitmap_height), Image.LANCZOS)
                else:
                    pixel_map = pixel_map.resize((bitmap_width, bitmap_height), Image.BICUBIC)
                pixel_map = pixel_map.tobytes()
                self.__bitmap = wx.Bitmap(bitmap_width, bitmap_height, 24)
                self.__bitmap.CopyFromBuffer(pixel_map, format=wx.BitmapBufferFormat_RGB)
                self.__bitmap_height = bitmap_height
                self.__bitmap_width = bitmap_width
                self.__ratio = ratio

        if self.__bitmap is not None:
            dc.DrawBitmap(self.__bitmap, 0, 0, True)

    def __paint_roi(self, dc):
        left = int(self.__roi.get_left() * self.__ratio)
        right = int(self.__roi.get_right() * self.__ratio)
        top = int(self.__roi.get_top() * self.__ratio)
        bottom = int(self.__roi.get_bottom() * self.__ratio)
        dc.SetPen(wx.Pen("red"))
        dc.SetBrush(wx.Brush(wx.Colour(0, 0, 0, wx.ALPHA_TRANSPARENT)))
        dc.DrawRectangle(left, top, right - left, bottom - top)

    def __left_mouse_click(self, evt):
        coord_x = int(evt.GetPosition().x / self.__ratio)
        coord_y = int(evt.GetPosition().y / self.__ratio)
        self.__dlg.define_roi_position(coord_x, coord_y)
