# -*- coding: utf-8

import wx
from .synchronization.selector import SynchronizationSelector
from .isolines.selector import IsolineSelector


class AccumulatorDlg(wx.Dialog):
    """
    This is the base class for all dialogs that create accumulators and set the accumulator properties
    """

    __train = None
    __synchronization_box = None
    __isoline_box = None

    __prefix_name_box = None
    __postfix_name_box = None
    __save_npz_box = None
    __add_to_manifest_box = None
    __save_mat_box = None
    __save_png_box = None

    def __init__(self, parent, train, title="Sample accumulator dialog"):
        super().__init__(parent, title=title, size=(800, 800))
        self.__train = train

        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)
        upper_panel = wx.BoxSizer(wx.HORIZONTAL)

        upper_left_panel = self.__create_synchronization_box(main_panel)
        upper_panel.Add(upper_left_panel, 1, wx.RIGHT, 5)

        upper_right_panel = wx.BoxSizer(wx.VERTICAL)

        isoline_box = self.__create_isoline_box(main_panel)
        upper_right_panel.Add(isoline_box, 0, wx.EXPAND | wx.BOTTOM, 5)

        accumulator_box = self._create_accumulator_box(main_panel)
        upper_right_panel.Add(accumulator_box, 0, wx.EXPAND | wx.BOTTOM, 5)

        output_box = self.__create_output_box(main_panel)
        upper_right_panel.Add(output_box, 0, wx.EXPAND)

        upper_panel.Add(upper_right_panel, 1, wx.EXPAND)
        main_layout.Add(upper_panel, 1, wx.EXPAND | wx.BOTTOM, 10)

        middle_box = self.__create_middle_box(main_panel)
        main_layout.Add(middle_box, 0, wx.EXPAND | wx.BOTTOM, 10)

        bottom_panel = self.__create_bottom_box(main_panel)
        main_layout.Add(bottom_panel, 0, wx.ALIGN_CENTRE)

        general_layout.Add(main_layout, 1, wx.EXPAND | wx.ALL, 10)
        main_panel.SetSizer(general_layout)
        self.Centre()

    def __create_synchronization_box(self, parent):
        self.__synchronization_box = SynchronizationSelector(parent, self.__train)
        return self.__synchronization_box

    def __create_isoline_box(self, parent):
        self.__isoline_box = IsolineSelector(parent, self.__train)
        return self.__isoline_box

    def _create_accumulator_box(self, parent):
        accumulator_box = wx.Panel(parent, size=(100, 200))
        accumulator_box.SetBackgroundColour("red")
        return accumulator_box

    def __create_output_box(self, parent):
        output_box = wx.StaticBoxSizer(wx.VERTICAL, parent, label="Output")
        layout = wx.BoxSizer(wx.VERTICAL)
        grid_sizer = wx.FlexGridSizer(2, 2, 5)
        grid_sizer.AddGrowableCol(1)

        prefix_name_caption = wx.StaticText(parent, label="Prefix name")
        grid_sizer.Add(prefix_name_caption, 0, wx.ALIGN_CENTER_VERTICAL)

        self.__prefix_name_box = wx.TextCtrl(parent)
        grid_sizer.Add(self.__prefix_name_box, 1, wx.EXPAND)

        postfix_name_caption = wx.StaticText(parent, label="Postfix name")
        grid_sizer.Add(postfix_name_caption, 0, wx.ALIGN_CENTER_VERTICAL)

        self.__postfix_name_box = wx.TextCtrl(parent)
        grid_sizer.Add(self.__postfix_name_box, 1, wx.EXPAND)

        layout.Add(grid_sizer, 1, wx.EXPAND | wx.BOTTOM, 5)

        self.__save_npz_box = wx.CheckBox(parent, label="Save to NPZ")
        self.__save_npz_box.Bind(wx.EVT_CHECKBOX, lambda event: self.__set_manifest_status())
        layout.Add(self.__save_npz_box, 0, wx.EXPAND, 5)

        add_to_manifest_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.__add_to_manifest_box = wx.CheckBox(parent, label="Add to manifest")
        add_to_manifest_sizer.Add(self.__add_to_manifest_box, 0, wx.LEFT, 20)
        layout.Add(add_to_manifest_sizer, 0, wx.BOTTOM, 5)
        self.__set_manifest_status()

        self.__save_mat_box = wx.CheckBox(parent, label="Save to MAT")
        layout.Add(self.__save_mat_box, 0, wx.BOTTOM, 5)

        self.__save_png_box = wx.CheckBox(parent, label="Save to PNG")
        layout.Add(self.__save_png_box)

        output_box.Add(layout, 1, wx.ALL | wx.EXPAND, 5)
        return output_box

    def __create_middle_box(self, parent):
        middle_box = wx.StaticText(parent,
                                   label="* LPF - low-pass filter, HPF - high-pass filter, BPF - band-pass filter")
        return middle_box

    def __create_bottom_box(self, parent):
        bottom_box = wx.BoxSizer(wx.HORIZONTAL)

        btn_ok = wx.Button(parent, label="OK")
        self.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_OK), btn_ok)
        bottom_box.Add(btn_ok, 0, wx.RIGHT, 5)

        btn_cancel = wx.Button(parent, label="Cancel")
        self.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_CANCEL), btn_cancel)
        bottom_box.Add(btn_cancel)

        return bottom_box

    def __set_manifest_status(self):
        if self.__save_npz_box.IsChecked():
            self.__add_to_manifest_box.Enable(True)
        else:
            self.__add_to_manifest_box.Enable(False)
            self.__add_to_manifest_box.SetValue(False)

    def close(self):
        del self.__train
        self.__synchronization_box.close()
        self.__isoline_box.close()

    def _get_accumulator_class(self):
        raise NotImplementedError("Attempt to use purely abstract class")

    def create_accumulator(self):
        synchronization = self.__synchronization_box.create_synchronization()
        isoline = self.__isoline_box.create_isoline(synchronization)
        AccumulatorClass = self._get_accumulator_class()
        accumulator = AccumulatorClass(isoline)
        return accumulator

    def get_train(self):
        return self.__train
