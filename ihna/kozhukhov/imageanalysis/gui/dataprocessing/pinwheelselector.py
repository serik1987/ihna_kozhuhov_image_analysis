# -*- coding: utf-8

from PIL import Image
import numpy as np
import wx
from ihna.kozhukhov.imageanalysis import PinwheelCenterList
from ihna.kozhukhov.imageanalysis.gui.pinwheelviewerdlg import PinwheelViewerDlg
from .datatodataprocessor import DataToDataProcessor


class PinwheelSelector(DataToDataProcessor):

    __draw_panel = None
    __coordinate_list = None
    __pinwheel_list_box = None
    __scale_box = None
    __x_position_box = None
    __y_position_box = None
    __local_map_height = None
    __local_map_width = None
    __bitmap_height = None
    __bitmap_width = None

    __phase_map = None
    __x_position = 0
    __y_position = 0
    __scale = 1.0
    __pinwheel_list = None

    __window_height = None
    __window_width = None
    __map_window_height = None
    __map_window_width = None

    def _get_default_minor_name(self):
        return "pinwheels"

    def _get_processor_title(self):
        return "Pinwheel selector"

    def _check_input_data(self):
        if self._input_data.is_phase_map():
            return
        raise ValueError("Please, select a phase map")

    def _place_additional_options(self, parent):
        self.__left_point = 0
        self.__top_point = 0
        self.__scale = 1.0
        self.__phase_map = self._input_data.get_data()
        self.__pinwheel_list = list()

        main_panel = wx.Panel(parent, size=(710, 600))
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        left_sizer = wx.BoxSizer(wx.VERTICAL)

        self.__window_height = 500
        self.__window_width = 500
        self.__draw_panel = wx.Panel(main_panel, size=(self.__window_width, self.__window_height))
        self.__draw_panel.SetBackgroundColour("green")
        left_sizer.Add(self.__draw_panel, 1, wx.EXPAND | wx.BOTTOM, 5)

        self.__coordinate_list = wx.StaticText(main_panel, label="X = undefined, Y = undefined")
        left_sizer.Add(self.__coordinate_list, 0, wx.BOTTOM, 5)

        hint = wx.StaticText(main_panel, label="Click to the pixel to add it to the list of pinwheel centers")
        left_sizer.Add(hint, 0)

        main_sizer.Add(left_sizer, 5, wx.EXPAND | wx.RIGHT, 10)
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        hint = wx.StaticText(main_panel, label="List of pinwheels")
        right_sizer.Add(hint, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__pinwheel_list_box = wx.ListBox(main_panel, choices=[],
                                              style=wx.LB_SINGLE | wx.LB_NEEDED_SB, size=(100, 300))
        right_sizer.Add(self.__pinwheel_list_box, 0, wx.EXPAND | wx.BOTTOM, 5)

        btn = wx.Button(main_panel, label="Delete pinwheel")
        btn.Bind(wx.EVT_BUTTON, lambda event: self.__delete_pinwheel())
        right_sizer.Add(btn, 0, wx.BOTTOM, 10)

        hint = wx.StaticText(main_panel, label="Scale")
        right_sizer.Add(hint, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__scale_box = wx.Slider(main_panel, value=1.0, minValue=100, maxValue=1000)
        self.__scale_box.Bind(wx.EVT_SCROLL, lambda event: self.__scroll_scale())
        right_sizer.Add(self.__scale_box, 0, wx.EXPAND | wx.BOTTOM, 10)

        hint = wx.StaticText(main_panel, label="X position, px")
        right_sizer.Add(hint, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__x_position_box = wx.Slider(main_panel, value=0, minValue=0, maxValue=0)
        self.__x_position_box.Bind(wx.EVT_SCROLL, lambda event: self.__scroll_position())
        right_sizer.Add(self.__x_position_box, 0, wx.EXPAND | wx.BOTTOM, 10)

        hint = wx.StaticText(main_panel, label="Y position, px")
        right_sizer.Add(hint, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__y_position_box = wx.Slider(main_panel, value=0, minValue=0, maxValue=0)
        self.__y_position_box.Bind(wx.EVT_SCROLL, lambda event: self.__scroll_position())
        right_sizer.Add(self.__y_position_box, 0, wx.EXPAND | wx.BOTTOM, 10)

        main_sizer.Add(right_sizer, 3, wx.EXPAND)
        main_panel.SetSizer(main_sizer)

        self.Bind(wx.EVT_SHOW, lambda event: self.__show())
        self.__draw_panel.Bind(wx.EVT_PAINT, lambda event: self.__paint())
        self.__draw_panel.Bind(wx.EVT_MOTION, self.__on_move)
        self.__draw_panel.Bind(wx.EVT_LEFT_DOWN, self.__add_pinwheel)
        return main_panel

    def __show(self):
        self.__update_pinwheel_list()
        self.__update_scale()
        self.__update_position()
        self.__update_image()

    def __update_pinwheel_list(self):
        pinwheel_lines = ["X = %d, Y = %d" % xy for xy in self.__pinwheel_list]
        self.__pinwheel_list_box.Set(pinwheel_lines)

    def __update_scale(self):
        pass

    def __update_position(self):
        win_size = self.__draw_panel.GetClientSize()
        self.__window_width = win_size.GetWidth()
        self.__window_height = win_size.GetHeight()
        self.__map_height, self.__map_width = self.__phase_map.shape
        if self.__window_height / self.__window_width >= self.__map_height / self.__map_width:
            self.__map_window_width = self.__map_height
            self.__map_window_height = self.__map_window_width * self.__window_height / self.__window_width
        else:
            self.__map_window_height = self.__map_height
            self.__map_window_width = self.__map_window_height * self.__window_width / self.__window_height
        self.__map_window_width = int(self.__map_window_width)
        self.__map_window_height = int(self.__map_window_height)
        x0_max = self.__map_width - self.__map_window_width / self.__scale
        if x0_max < 0:
            x0_max = 0
        x0_max = int(round(x0_max))
        y0_max = self.__map_height - self.__map_window_height / self.__scale
        if y0_max < 0:
            y0_max = 0
        y0_max = int(round(y0_max))
        self.__x_position_box.SetMax(x0_max)
        self.__y_position_box.SetMax(y0_max)

    def __update_image(self):
        self.Refresh()

    def __add_pinwheel(self, event):
        x_map, y_map = self.__get_map_coordinates(event)
        self.__pinwheel_list.append((x_map, y_map))
        self.__update_pinwheel_list()
        self.__update_image()

    def __delete_pinwheel(self):
        idx = self.__pinwheel_list_box.GetSelection()
        if idx == wx.NOT_FOUND:
            idx = -1
        if len(self.__pinwheel_list) > 0:
            self.__pinwheel_list.pop(idx)
        self.__update_pinwheel_list()
        self.__update_image()

    def __scroll_scale(self):
        self.__scale = float(self.__scale_box.GetValue()) / 100.0
        self.__update_position()
        self.__update_image()

    def __scroll_position(self):
        self.__update_image()

    def __paint(self):
        x_min = self.__x_position_box.GetValue()
        y_min = self.__y_position_box.GetValue()
        self.__x_position = x_min
        self.__y_position = y_min
        x_max = round(x_min + self.__map_window_width / self.__scale)
        y_max = round(y_min + self.__map_window_height / self.__scale)
        local_map = self.__phase_map[y_min:y_max, x_min:x_max]
        dc = wx.PaintDC(self.__draw_panel)
        brush = wx.Brush("white")
        dc.SetBackground(brush)
        dc.Clear()
        self.__paint_image(dc, local_map)
        self.__paint_pinwheels(dc)

    def __paint_image(self, dc, image):
        initial_height, initial_width = image.shape
        final_height, final_width = self.__window_height, self.__window_width
        height_ratio = final_height / initial_height
        width_ratio = final_width / initial_width
        ratio = min(height_ratio, width_ratio)
        bitmap_height = int(initial_height * ratio)
        bitmap_width = int(initial_width * ratio)
        harm = self._input_data.get_harmonic()
        hue_map = image * harm * 128 / np.pi
        hue_map = np.array(hue_map, dtype=np.uint8)
        saturation_map = np.ones((initial_height, initial_width), dtype=np.uint8) * 255
        value_map = saturation_map.copy()
        raw_pixel_map = np.zeros((initial_height, initial_width, 3), dtype=np.uint8)
        raw_pixel_map[:, :, 0] = hue_map
        raw_pixel_map[:, :, 1] = saturation_map
        raw_pixel_map[:, :, 2] = value_map
        pixel_map = Image.fromarray(raw_pixel_map, "HSV")
        pixel_map = pixel_map.convert(mode="RGB")
        if ratio <= 1.0:
            pixel_map = pixel_map.resize((bitmap_width, bitmap_height), Image.LANCZOS)
        else:
            pixel_map = pixel_map.resize((bitmap_width, bitmap_height), Image.BICUBIC)
        pixel_map = pixel_map.tobytes()
        bitmap = wx.Bitmap(bitmap_width, bitmap_height, 24)
        bitmap.CopyFromBuffer(pixel_map, format=wx.BitmapBufferFormat_RGB)
        dc.DrawBitmap(bitmap, 0, 0, True)
        self.__local_map_height = initial_height
        self.__local_map_width = initial_width
        self.__bitmap_height = bitmap_height
        self.__bitmap_width = bitmap_width

    def __paint_pinwheels(self, dc):
        for pwc in self.__pinwheel_list:
            x_map = pwc[0]
            y_map = pwc[1]
            x = round((x_map - self.__x_position) * self.__bitmap_width / self.__local_map_width)
            y = round((y_map - self.__y_position) * self.__bitmap_height / self.__local_map_height)
            if 0 <= x < self.__bitmap_width and 0 <= y < self.__bitmap_height:
                width = round(self.__bitmap_width / self.__local_map_width)
                height = round(self.__bitmap_height / self.__local_map_height)
                if width < 1:
                    width = 1
                if height < 1:
                    height = 1
                dc.DrawRectangle(x, y, width, height)

    def __get_map_coordinates(self, event):
        x_win = event.GetPosition().x
        y_win = event.GetPosition().y
        if 0 <= x_win < self.__bitmap_width and 0 <= y_win < self.__bitmap_height:
            x_map = int(self.__x_position + x_win * self.__local_map_width / self.__bitmap_width)
            y_map = int(self.__y_position + y_win * self.__local_map_height / self.__bitmap_height)
        else:
            x_map = None
            y_map = None
        return x_map, y_map

    def __on_move(self, event):
        x_map, y_map = self.__get_map_coordinates(event)
        if x_map is None:
            return
        info = "X = %d, Y = %d" % (x_map, y_map)
        self.__coordinate_list.SetLabel(info)

    def _process(self):
        features = self._input_data.get_features().copy()
        features['minor_name'] = self.get_output_file()
        features['original_map'] = self._input_data.get_full_name()
        features['is_main'] = 'no'
        data = self.__pinwheel_list, self.__phase_map.shape[1], self.__phase_map.shape[0]
        self._output_data = PinwheelCenterList(features, data)

    def _get_result_viewer(self):
        return PinwheelViewerDlg
