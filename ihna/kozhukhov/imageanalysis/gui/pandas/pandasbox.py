# -*- coding: utf-8

import wx
import pandas as pd
from .PandasColumnEditorDlg import PandasColumnEditorDlg


class PandasBox(wx.Dialog):

    __animal_list = None
    __column_descriptions = None

    __btn_add_column = None
    __btn_edit_column = None
    __btn_delete_column = None
    __btn_save = None

    __column_list = None
    __column_list_box = None

    def __init__(self, parent, animal_list):
        self.__animal_list = animal_list

        super().__init__(parent, title="Create table", size=(400, 600))
        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)
        upper_layout = wx.BoxSizer(wx.HORIZONTAL)
        left_layout = wx.BoxSizer(wx.VERTICAL)

        column_caption = wx.StaticText(main_panel, label="Columns")
        left_layout.Add(column_caption, 0, wx.BOTTOM, 5)

        self.__column_list_box = wx.ListBox(main_panel,
                                            style=wx.LB_SINGLE)
        left_layout.Add(self.__column_list_box, 1, wx.EXPAND)

        upper_layout.Add(left_layout, 1, wx.EXPAND | wx.RIGHT, 5)
        right_layout = wx.BoxSizer(wx.VERTICAL)

        self.__btn_add_column = wx.Button(main_panel, label="Add column")
        self.__btn_add_column.Bind(wx.EVT_BUTTON, lambda event: self.add_column())
        right_layout.Add(self.__btn_add_column, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__btn_edit_column = wx.Button(main_panel, label="Change column")
        self.__btn_edit_column.Bind(wx.EVT_BUTTON, lambda event: self.edit_column())
        right_layout.Add(self.__btn_edit_column, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__btn_delete_column = wx.Button(main_panel, label="Delete column")
        self.__btn_delete_column.Bind(wx.EVT_BUTTON, lambda event: self.delete_column())
        right_layout.Add(self.__btn_delete_column, 0, wx.EXPAND | wx.BOTTOM, 5)

        upper_layout.Add(right_layout, 0, wx.EXPAND | wx.TOP, 20)
        main_layout.Add(upper_layout, 1, wx.EXPAND | wx.BOTTOM, 5)
        lower_layout = wx.BoxSizer(wx.HORIZONTAL)

        save_button = wx.Button(main_panel, label="Save")
        save_button.Bind(wx.EVT_BUTTON, lambda event: self.create_pandas_table())
        lower_layout.Add(save_button, 0, wx.RIGHT, 5)

        cancel_button = wx.Button(main_panel, label="Close")
        cancel_button.Bind(wx.EVT_BUTTON, lambda event: self.Close())
        lower_layout.Add(cancel_button, 0)

        main_layout.Add(lower_layout, 0, wx.ALIGN_CENTER)
        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizer(general_layout)
        self.Centre()

        self.__btn_save = save_button
        self.load_all_columns()

    def create_pandas_table(self):
        try:
            dlg = wx.FileDialog(self,
                                message="Save result table to...",
                                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                                wildcard="CSV file|*.csv|XLS file|*.xls;*.xlsx|JSON file|*.json|HTML file|*.html")
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            filename = dlg.GetPath()
            filter_index = dlg.GetFilterIndex()
            df_source = {}
            for column in self.__column_list:
                print("Saving", column.get_name())
                df_source[column.get_name()] = column.get_values(self.__animal_list)
            df = pd.DataFrame(df_source)
            if filter_index == 0:
                if not filename.endswith(".csv"):
                    filename += ".csv"
                df.to_csv(filename)
            if filter_index == 1:
                if not filename.endswith(".xlsx") and not filename.endswith(".xls"):
                    filename += ".xlsx"
                df.to_excel(filename)
            if filter_index == 2:
                if not filename.endswith(".json"):
                    filename += ".json"
                df.to_json(filename)
            if filter_index == 3:
                if not filename.endswith(".html"):
                    filename += ".html"
                df.to_html(filename)
        except Exception as err:
            from ihna.kozhukhov.imageanalysis.gui import MainWindow
            MainWindow.show_error_message(self, err, "Create result table")

    def add_column(self):
        try:
            dlg = PandasColumnEditorDlg(self)
            if dlg.ShowModal() == wx.ID_OK:
                column = dlg.get_column()
                if self.__column_list is None:
                    self.__column_list = []
                self.__column_list.append(column)
                self.load_all_columns()
        except Exception as err:
            from ihna.kozhukhov.imageanalysis.gui import MainWindow
            MainWindow.show_error_message(self, err, "Add column")

    def edit_column(self):
        try:
            column = self.get_selected_column()
            dlg = PandasColumnEditorDlg(self, column)
            if dlg.ShowModal() == wx.ID_OK:
                column = dlg.get_column()
                self.__column_list[self.__column_list_box.GetSelection()] = column
                self.load_all_columns()
        except Exception as err:
            from ihna.kozhukhov.imageanalysis.gui import MainWindow
            MainWindow.show_error_message(self, err, "Change column")

    def delete_column(self):
        try:
            column = self.get_selected_column()
            self.__column_list.remove(column)
            self.load_all_columns()
        except Exception as err:
            from ihna.kozhukhov.imageanalysis.gui import MainWindow
            MainWindow.show_error_message(self, err, "Delete column")

    def load_all_columns(self):
        if self.__column_list is None or len(self.__column_list) == 0:
            self.__column_list_box.Clear()
            self.__btn_edit_column.Enable(False)
            self.__btn_delete_column.Enable(False)
            self.__btn_save.Enable(False)
        else:
            self.__btn_edit_column.Enable(True)
            self.__btn_delete_column.Enable(True)
            self.__btn_save.Enable(True)
            column_names = [column.get_name() for column in self.__column_list]
            self.__column_list_box.SetItems(column_names)
            self.__column_list_box.SetSelection(0)

    def get_selected_column(self):
        selection = self.__column_list_box.GetSelection()
        return self.__column_list[selection]
