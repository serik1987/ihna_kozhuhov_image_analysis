# -*- coding: utf-8

from time import sleep
import numpy as np
import wx
from ihna.kozhukhov.imageanalysis import ImagingMap


class DataProcessDlg(wx.Dialog):

    _considering_case = None
    _input_data = None
    _parent = None
    _output_data = None

    __output_file_box = None
    __save_to_npz_box = None
    __add_to_manifest_box = None
    __save_to_mat_box = None

    __value_box = None
    __add_to_features = None

    __prefix_box = None
    __postfix_box = None

    def __init__(self, parent, input_data, considering_case):
        self._input_data = input_data
        self._parent = parent
        self._considering_case = considering_case
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

    def _place_general_options(self, parent):
        raise NotImplementedError("DataProcessDlg._place_general_options(parent)")

    def _place_additional_options(self, parent):
        raise NotImplementedError("DataProcessDlg._place_additional_options(parent)")

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
        save_details.Add(self.__save_to_mat_box, 0, wx.EXPAND | wx.BOTTOM, 0)

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
        value_save_details.Add(self.__add_to_features, 0, wx.EXPAND | wx.BOTTOM, 0)

        self.__value_box = value_box
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

    def get_output_file(self):
        if self.__output_file_box is None:
            raise AttributeError("The output file box is not added by _place_output_file_box")
        else:
            return self.__output_file_box.GetValue()

    def __continue_processing(self):
        try:
            progress_dlg = wx.ProgressDialog(self._get_processor_title(),
                                             self._get_processor_title() + " is going on...",
                                             maximum=100,
                                             parent=self)
            progress_dlg.Update(0)
            try:
                self._process()
                self._save_processed_data()
            except Exception as err:
                progress_dlg.Destroy()
                raise err
            progress_dlg.Destroy()
            self.EndModal(wx.ID_OK)
        except Exception as err:
            from ihna.kozhukhov.imageanalysis.gui import MainWindow
            MainWindow.show_error_message(self, err, self._get_processor_title())

    def _process(self):
        raise NotImplementedError("DataProcessDlg._process()")

    def _save_processed_data(self):
        raise NotImplementedError("DataProcessDlg._save_processed_data()")

    def get_output_data(self):
        return self._output_data

    def _save_output_data(self):
        self._output_data.get_features()['minor_name'] = self.get_output_file()
        folder_name = self._considering_case['pathname']
        result_dlg = self._get_result_viewer()(self, self._output_data)
        result_dlg.ShowModal()
        if self.is_save_npz_selected():
            self._output_data.save_npz(folder_name)
            if self.is_add_to_manifest_selected():
                self._considering_case.add_data(self._output_data)
        if self.is_save_mat_selected():
            self._output_data.save_mat(folder_name)
        if self.is_save_png_selected():
            result_dlg.save_png(folder_name)

    def is_save_npz_selected(self):
        if self.__save_to_npz_box is None:
            raise AttributeError("To enable this option please, apply _place_save_details method")
        return self.__save_to_npz_box.IsChecked()

    def is_add_to_manifest_selected(self):
        if self.__add_to_manifest_box is None:
            raise AttributeError("To enable this option please, apply _place_save_details method")
        return self.__save_to_npz_box.IsChecked() and self.__add_to_manifest_box.IsChecked()

    def is_save_mat_selected(self):
        if self.__save_to_mat_box is None:
            raise AttributeError("To enable this option please, apply _place_save_details method")
        return self.__save_to_mat_box.IsChecked()

    def is_save_png_selected(self):
        if self.__save_to_png is None:
            raise AttributeError("To enable this option please, apply _place_save_details method")
        return self.__save_to_png.IsChecked()

    def is_add_to_map_features(self):
        if self.__add_to_features is None:
            raise AttributeError("To enable this option please, place _value_save_details method")
        return self.__add_to_features.IsChecked()

    def get_value_key(self):
        if self.__value_box is None:
            raise AttributeError("To enable this option please, place _value_save_details method")
        return self.__value_box.GetValue()

    def _get_result_viewer(self):
        raise NotImplementedError("DataProcessDlg._get_result_viewer()")
