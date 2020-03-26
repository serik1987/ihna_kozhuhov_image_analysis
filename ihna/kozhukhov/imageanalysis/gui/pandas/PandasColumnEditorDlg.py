# -*- coding: utf-8

import wx
from .pandasfieldcolumn import PandasFieldColumn
from .pandasfeaturecolumn import PandasFeatureColumn
from .pandasfieldcolumneditor import PandasFieldColumnEditor
from .pandasfeaturecolumneditor import PandasFeatureColumnEditor


class PandasColumnEditorDlg(wx.Dialog):

    __current_column = None
    __internal_classes = [PandasFieldColumn, PandasFeatureColumn]
    __column_classes = [PandasFieldColumnEditor, PandasFeatureColumnEditor]
    __column_editors = None
    __column_names = ["From electronic journal", "From data"]
    __column_selectors = None

    __edt_column_name = None

    __main_panel = None

    def _get_title(self):
        if self.__current_column is None:
            return "Add column"
        else:
            return "Change column"

    def _get_do_button_caption(self):
        if self.__current_column is None:
            return "Add"
        else:
            return "Edit"

    def __init__(self, parent, column=None):
        """
        How to create:
            dlg = PandasColumnEditorDlg(parent) # creates 'Add column' dlg
            dlg = PandasColumnEditorDlg(parent, column) # creates 'Change column' dlg
        """
        self.__current_column = column
        super().__init__(parent, title=self._get_title(), size=(400, 500))
        main_panel = wx.Panel(self)
        general_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        column_name_layout = wx.BoxSizer(wx.HORIZONTAL)
        column_name_caption = wx.StaticText(main_panel, label="Column name")
        column_name_layout.Add(column_name_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        self.__edt_column_name = wx.TextCtrl(main_panel)
        column_name_layout.Add(self.__edt_column_name, 1, wx.EXPAND)
        main_sizer.Add(column_name_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

        column_names_layout = wx.BoxSizer(wx.HORIZONTAL)
        column_style = wx.RB_GROUP
        left_border = 0
        self.__column_selectors = []
        for column_name in self.__column_names:
            column_selector = wx.RadioButton(main_panel, label=column_name, style=column_style)
            column_selector.Bind(wx.EVT_RADIOBUTTON, lambda event: self.select_column_editor())
            column_names_layout.Add(column_selector, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, left_border)
            self.__column_selectors.append(column_selector)
            column_style = 0
            left_border = 5
        main_sizer.Add(column_names_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

        all_columns_sizer = wx.BoxSizer(wx.VERTICAL)
        self.__column_editors = []
        for column_class in self.__column_classes:
            try:
                column_editor = column_class(main_panel, column)
            except Exception as err:
                column_editor = wx.StaticText(main_panel, label=str(err))
            column_editor.Hide()
            all_columns_sizer.Add(column_editor, 0, wx.BOTTOM | wx.EXPAND, 5)
            self.__column_editors.append(column_editor)

        main_sizer.Add(all_columns_sizer, 1, wx.EXPAND)
        buttons_sizer = wx.BoxSizer(wx.HORIZONTAL)

        do_button = wx.Button(main_panel, label=self._get_do_button_caption())
        do_button.Bind(wx.EVT_BUTTON, lambda event: self.check())
        buttons_sizer.Add(do_button, 0, wx.RIGHT, 5)

        cancel_button = wx.Button(main_panel, label="Cancel")
        cancel_button.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_CANCEL))
        buttons_sizer.Add(cancel_button)

        main_sizer.Add(buttons_sizer, 0, wx.ALIGN_CENTER)
        general_sizer.Add(main_sizer, 1, wx.EXPAND | wx.ALL, 10)
        main_panel.SetSizer(general_sizer)
        self.Centre()

        self.__main_panel = main_panel

        if column is not None:
            self.set_column()
        self.select_column_editor()

    def select_column_editor(self):
        idx = 0
        for editor in self.__column_editors:
            editor.Hide()
        for column_selector in self.__column_selectors:
            if column_selector.GetValue():
                editor = self.__column_editors[idx]
                editor.Show()
                self.__main_panel.Layout()
                self.Layout()
                return
            idx += 1

    def get_selected_editor(self):
        idx = 0
        for column_selector in self.__column_selectors:
            if column_selector.GetValue():
                return self.__column_editors[idx]
            idx += 1

    def check(self):
        try:
            if self.__edt_column_name.GetValue() == "":
                raise ValueError("Please, specify a column name")
            self.get_selected_editor().additional_check()
        except Exception as err:
            from ihna.kozhukhov.imageanalysis.gui import MainWindow
            MainWindow.show_error_message(self, err, self._get_title())
            return
        self.EndModal(wx.ID_OK)

    def get_column(self):
        editor = self.get_selected_editor()
        column_name = self.__edt_column_name.GetValue()
        column = editor.get_column(column_name)
        return column

    def set_column(self):
        self.__edt_column_name.SetValue(self.__current_column.get_name())
        idx = 0
        for column_class in self.__internal_classes:
            if isinstance(self.__current_column, column_class):
                break
            idx += 1
        self.__column_selectors[idx].SetValue(True)
        self.__column_editors[idx].set_column()
