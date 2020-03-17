# -*- coding: utf-8

import numpy as np
import wx
from ihna.kozhukhov.imageanalysis import ImagingMap


class DataProcessDlg(wx.Dialog):

    _input_data = None
    _parent = None
    __output_data = None

    __output_file_box = None
    __save_to_npz_box = None
    __add_to_manifest_box = None
    __save_to_mat_box = None

    __value_caption = None
    __add_to_features = None

    __prefix_box = None
    __postfix_box = None

    def __init__(self, parent, input_data):
        self._input_data = input_data
        self._parent = parent
        self._check_input_data()
        super().__init__(parent,
                         title="%s: %s" % (self._get_processor_title(), input_data.get_full_name()),
                         size=(800, 500))
        main_panel = wx.Panel(self)
        general_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        general_options = self._place_general_options(main_panel)
        additional_options = self._place_additional_options(main_panel)
        if additional_options is None:
            main_sizer.Add(general_options, 0, wx.EXPAND | wx.BOTTOM, 10)
        else:
            main_sizer.Add(general_options, 0, wx.BOTTOM | wx.EXPAND, 5)
            main_sizer.Add(additional_options, 0, wx.BOTTOM | wx.EXPAND, 10)

        lower_sizer = wx.BoxSizer(wx.HORIZONTAL)

        btn_ok = wx.Button(main_panel, label="OK")
        btn_ok.Bind(wx.EVT_BUTTON, lambda event: self.__continue_processing())
        lower_sizer.Add(btn_ok, 0, wx.RIGHT, 5)

        btn_cancel = wx.Button(main_panel, label="Cancel")
        btn_cancel.Bind(wx.EVT_BUTTON, lambda event: self.Close())
        lower_sizer.Add(btn_cancel)

        main_sizer.Add(lower_sizer, 0, wx.ALIGN_CENTER)
        general_sizer.Add(main_sizer, 1, wx.EXPAND | wx.ALL, 10)
        main_panel.SetSizerAndFit(general_sizer)
        self.Centre()
        self.Fit()

    def _get_processor_title(self):
        return "Sample processor"

    def _get_default_minor_name(self):
        raise NotImplementedError("DataProcessDlg._get_default_minor_name()")

    def _check_input_data(self):
        raise NotImplementedError("Unable to use abstract data processor")

    def _check_imaging_map(self, complex_warn=False):
        if not isinstance(self._input_data, ImagingMap):
            raise ValueError("The average data can be computed for maps only")
        if complex_warn and self._input_data.get_data().dtype == np.complex:
            dlg = wx.MessageDialog(self._parent,
                                   "The average value for complex maps is some meaningless complex value",
                                   "Map average",
                                   wx.OK | wx.CENTRE | wx.ICON_WARNING)
            dlg.ShowModal()

    def __continue_processing(self):
        print("CONTINUE PROCESSING")

    def _place_general_options(self, parent):
        raise NotImplementedError("DataProcessDlg._place_general_options(parent)")

    def _place_additional_options(self, parent):
        additional_options = wx.Panel(parent, size=(200, 30))
        additional_options.SetBackgroundColour("blue")

        return additional_options

    def _place_output_file_box(self, parent):
        output_file_layout = wx.BoxSizer(wx.HORIZONTAL)

        output_file_caption = wx.StaticText(parent, label="Minor name of the output map")
        output_file_layout.Add(output_file_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        self.__output_file_box = wx.TextCtrl(parent, value=self._get_default_minor_name())
        output_file_layout.Add(self.__output_file_box, 1, wx.EXPAND)

        return output_file_layout

    def __set_save_npz(self):
        if self.__save_to_npz_box.IsChecked():
            self.__add_to_manifest_box.Enable(True)
        else:
            self.__add_to_manifest_box.Enable(False)
            self.__add_to_manifest_box.SetValue(False)

    def _place_save_details(self, parent):
        save_details = wx.BoxSizer(wx.VERTICAL)

        self.__save_to_npz_box = wx.CheckBox(parent, label="Save to NPZ")
        self.__save_to_npz_box.Bind(wx.EVT_CHECKBOX, lambda event: self.__set_save_npz())
        save_details.Add(self.__save_to_npz_box, 0, wx.EXPAND | wx.BOTTOM, 5)

        add_to_manifest_layout = wx.BoxSizer(wx.HORIZONTAL)
        self.__add_to_manifest_box = wx.CheckBox(parent, label="Add to manifest")
        self.__add_to_manifest_box.Enable(False)
        add_to_manifest_layout.Add(self.__add_to_manifest_box, 0, wx.LEFT | wx.EXPAND, 20)
        save_details.Add(add_to_manifest_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__save_to_mat_box = wx.CheckBox(parent, label="Save to MAT")
        save_details.Add(self.__save_to_mat_box, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__save_to_png = wx.CheckBox(parent, label="Save to PNG")
        save_details.Add(self.__save_to_png, 0, wx.EXPAND)

        return save_details

    def _place_value_save_details(self, parent):
        value_save_details = wx.BoxSizer(wx.VERTICAL)

        value_layout = wx.BoxSizer(wx.HORIZONTAL)
        value_caption = wx.StaticText(parent, label="Value key")
        value_layout.Add(value_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        value_box = wx.TextCtrl(parent, value=self._get_default_minor_name())
        value_layout.Add(value_box, 1, wx.EXPAND)
        value_save_details.Add(value_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__add_to_features = wx.CheckBox(parent, label="Add to map features")
        value_save_details.Add(self.__add_to_features, 0, wx.EXPAND | wx.BOTTOM, 5)

        return value_save_details

    def _place_major_name(self, parent):
        major_name = wx.BoxSizer(wx.VERTICAL)

        prefix_name_layout = wx.BoxSizer(wx.HORIZONTAL)
        prefix_name_caption = wx.StaticText(parent, label="Prefix name")
        prefix_name_layout.Add(prefix_name_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.__prefix_box = wx.TextCtrl(parent)
        prefix_name_layout.Add(self.__prefix_box, 1, wx.EXPAND)
        major_name.Add(prefix_name_layout, 0, wx.BOTTOM | wx.EXPAND, 5)

        postfix_name_layout = wx.BoxSizer(wx.HORIZONTAL)
        postfix_name_caption = wx.StaticText(parent, label="Postfix name")
        postfix_name_layout.Add(postfix_name_caption, 0, wx.ALIGN_CENTER | wx.RIGHT, 5)
        self.__postfix_box = wx.TextCtrl(parent)
        postfix_name_layout.Add(self.__postfix_box, 1, wx.EXPAND)
        major_name.Add(postfix_name_layout, 0, wx.EXPAND)

        return major_name
