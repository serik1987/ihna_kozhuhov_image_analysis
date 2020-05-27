# -*- coding: utf-8

import wx
from .dataprocessing import get_data_processors


class ResultListDlg(wx.Dialog):

    __case = None

    __map_list = None
    __map_processors_list = None
    __all_dialogs = None

    def __init__(self, parent, case):
        self.__case = case
        super().__init__(parent, title=self.get_title(), size=(700, 500))
        main_panel = wx.Panel(self)
        general_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        upper_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.__map_list = wx.ListBox(main_panel, choices=self.get_data_names(),
                                     style=wx.LB_SINGLE | wx.LB_NEEDED_SB | wx.LB_SORT)
        upper_sizer.Add(self.__map_list, 1, wx.EXPAND | wx.RIGHT, 10)

        self.__all_dialogs = get_data_processors(self)
        choices = list(self.__all_dialogs.keys())
        self.__map_processors_list = wx.ListBox(main_panel, choices=choices,
                                                style=wx.LB_SINGLE | wx.LB_NEEDED_SB | wx.LB_SORT)
        upper_sizer.Add(self.__map_processors_list, 1, wx.EXPAND)

        main_sizer.Add(upper_sizer, 1, wx.EXPAND | wx.BOTTOM, 10)
        middle_sizer = wx.BoxSizer(wx.HORIZONTAL)

        view_button = wx.Button(main_panel, label="View")
        view_button.Bind(wx.EVT_BUTTON, lambda event: self.view_map())
        middle_sizer.Add(view_button, 0, wx.RIGHT, 5)

        delete_button = wx.Button(main_panel, label="Delete")
        delete_button.Bind(wx.EVT_BUTTON, lambda event: self.delete_map())
        middle_sizer.Add(delete_button, 0, wx.RIGHT, 5)

        process_button = wx.Button(main_panel, label="Process")
        process_button.Bind(wx.EVT_BUTTON, lambda event: self.process())
        middle_sizer.Add(process_button, 0)

        main_sizer.Add(middle_sizer, 0, wx.EXPAND)
        general_sizer.Add(main_sizer, 1, wx.EXPAND | wx.ALL, 10)
        main_panel.SetSizer(general_sizer)
        self.Centre()

    def get_case(self):
        return self.__case

    def get_title(self):
        return "%s: %s_%s" % (self._get_base_title(), self.__case.get_animal_name(), self.__case['short_name'])

    def _get_base_title(self):
        raise NotImplementedError("_get_base_title")

    def view_map(self):
        try:
            data = self.__case.get_data(self.get_map_name())
            data.load_data()
            data_dlg = self._create_map_viewer_dlg(data)
            data_dlg.ShowModal()
        except Exception as err:
            from .MainWindow import MainWindow
            MainWindow.show_error_message(self, err, "Map view")

    def delete_map(self):
        try:
            map_name = self.get_map_name()
            self.__case.delete_data(map_name)
            self.__map_list.SetItems(self.get_data_names())
        except Exception as err:
            from .MainWindow import MainWindow
            MainWindow.show_error_message(self, err, "Delete map")

    def process(self):
        try:
            from ihna.kozhukhov.imageanalysis.mapprocessing import spatial_filter
            from .complexmapviewerdlg import ComplexMapViewerDlg
            input_data = self.__case.get_data(self.get_map_name())
            if input_data is None:
                raise ValueError("Please, select an appropriate data from the left list")
            input_data.load_data()
            input_data_dlg_selection = self.__map_processors_list.GetStringSelection()
            if input_data_dlg_selection == "":
                raise ValueError("Please, select an appropriate processor from list on the right")
            input_data_dlg = self.__all_dialogs[input_data_dlg_selection](self, input_data, self.__case)
            if input_data_dlg.get_input_data() is None:
                return
            input_data_dlg.ShowModal()
            self.__map_list.SetItems(self.get_data_names())
        except Exception as err:
            from .MainWindow import MainWindow
            MainWindow.show_error_message(self, err, "Process map data")

    def get_data_names(self):
        data_names = []
        for data in self.__case.data():
            if data.__class__ == self.get_data_class():
                data_names.append(data.get_full_name())
        return data_names

    def get_data_class(self):
        raise NotImplementedError("get_data_class")

    def get_map_name(self):
        return self.__map_list.GetStringSelection()

    def _create_map_viewer_dlg(self, data):
        raise NotImplementedError("_create_map_viewer_dlg")
