# -*- coding: utf-8

import os
import time
import wx
import ihna.kozhukhov.imageanalysis.manifest as manifest
import ihna.kozhukhov.imageanalysis.sourcefiles as sfiles
from .importcasedialog import ImportCaseDialog
from .importcasemanager import ImportCaseManager
from .nativedatamanager import NativeDataManager
from .roimanager import RoiManager
from .animalfilterdlg import AnimalFilterDlg
from .casefilterdlg import CaseFilterDlg
from .autoprocess import *
from .tracesresultlistdlg import TraceResultListDlg
from .mapresultlistdlg import MapResultListDlg
from ihna.kozhukhov.imageanalysis.gui.pandas.pandasbox import PandasBox


class MainWindow(wx.Frame):
    """
    This is the main window of the application
    """

    __animals_box = None
    __new_animal = None
    __delete_animal = None
    __animal_filter = None
    __specimen_box = None
    __conditions_box = None
    __recording_site_box = None
    __save_animal_info = None
    __cases_box = None
    __import_case = None
    __delete_case = None
    __case_filter = None
    __case_short_name_box = None
    __case_long_name_box = None
    __stimulation_box = None
    __additional_stimulation_box = None
    __special_conditions_box = None
    __additional_information_box = None
    __save_case_info = None
    __case_info_label = None
    __native_data_label = None
    __native_data_manager = None
    __roi_label = None
    __roi_data_manager = None
    __trace_analysis_label = None
    __trace_analysis_manager = None
    __averaged_maps_label = None
    __averaged_maps_manager = None
    __compressed_state = None
    __roi_exist = None
    __include_auto_box = None
    __decompress_before_processing = False
    __decompress_before_processing_box = None
    __extract_frame_button = None
    __trace_analysis_button = None
    __auto_average_maps_box = None
    __working_dir = None

    __animals = None
    __animal = None
    __cases = None
    __case = None

    def __init__(self):
        super().__init__(None, title="Image Analysis", size=(1000, 700),
                         style=wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX))
        panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.HORIZONTAL)
        main_layout = wx.BoxSizer(wx.HORIZONTAL)

        left_panel = self.__create_left_panel(panel)
        main_layout.Add(left_panel, 3, wx.RIGHT | wx.EXPAND, 5)

        middle_panel = self.__create_middle_panel(panel)
        main_layout.Add(middle_panel, 3, wx.RIGHT | wx.EXPAND, 5)

        right_panel = self.__create_right_panel(panel)
        main_layout.Add(right_panel, 3, wx.EXPAND, 0)

        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        panel.SetSizer(general_layout)
        self.Centre(wx.BOTH)

    def __create_left_panel(self, panel):
        left_panel = wx.BoxSizer(wx.VERTICAL)
        left_panel_caption = wx.StaticText(panel, label="Animals", style=wx.ALIGN_LEFT)
        left_panel.Add(left_panel_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__animals_box = wx.ListBox(panel, size=(100, 150), style=wx.LB_SINGLE | wx.LB_NEEDED_SB | wx.LB_SORT)
        self.Bind(wx.EVT_LISTBOX, self.select_animal, self.__animals_box)
        left_panel.Add(self.__animals_box, 0, wx.BOTTOM | wx.EXPAND, 5)
        left_button_panel = wx.BoxSizer(wx.HORIZONTAL)

        self.__new_animal = wx.Button(panel, label="New animal")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.new_animal(), self.__new_animal)
        left_button_panel.Add(self.__new_animal, 0, wx.RIGHT, 5)

        self.__delete_animal = wx.Button(panel, label="Delete animal")
        self.__delete_animal.Enable(False)
        self.Bind(wx.EVT_BUTTON, lambda evt: self.delete_animal(), self.__delete_animal)
        left_button_panel.Add(self.__delete_animal, 0, wx.RIGHT, 5)

        self.__animal_filter = wx.Button(panel, label="Animal filter")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.open_animal_filter(), self.__animal_filter)
        left_button_panel.Add(self.__animal_filter, 0, 0, 0)

        left_panel.Add(left_button_panel, 0, wx.BOTTOM, 5)
        left_box = wx.StaticBox(panel, label="Animal info", style=wx.ALIGN_LEFT)
        left_panel.Add(left_box, 1, wx.EXPAND, 0)

        left_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        left_panel_content_sizer = wx.BoxSizer(wx.VERTICAL)
        specimen_box_caption = wx.StaticText(left_box, label="Specimen")
        left_panel_content_sizer.Add(specimen_box_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__specimen_box = wx.TextCtrl(left_box, value="")
        left_panel_content_sizer.Add(self.__specimen_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        conditions_caption = wx.StaticText(left_box, label="Conditions")
        left_panel_content_sizer.Add(conditions_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__conditions_box = wx.TextCtrl(left_box)
        left_panel_content_sizer.Add(self.__conditions_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        recording_site_label = wx.StaticText(left_box, label="Recording site")
        left_panel_content_sizer.Add(recording_site_label, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__recording_site_box = wx.TextCtrl(left_box)
        left_panel_content_sizer.Add(self.__recording_site_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__save_animal_info = wx.Button(left_box, label="Save animal info")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.save_animal_info(), self.__save_animal_info)
        left_panel_content_sizer.Add(self.__save_animal_info, 0, 0, 0)

        left_panel_sizer.Add(left_panel_content_sizer, 1, wx.ALL | wx.EXPAND, 5)
        left_box.SetSizer(left_panel_sizer)
        return left_panel

    def __create_middle_panel(self, panel):
        middle_panel = wx.BoxSizer(wx.VERTICAL)
        middle_panel_caption = wx.StaticText(panel, label="Cases")
        middle_panel.Add(middle_panel_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__cases_box = wx.ListBox(panel, size=(100, 150), style=wx.LB_SINGLE | wx.LB_NEEDED_SB | wx.LB_SORT)
        self.__cases_box.Bind(wx.EVT_LISTBOX, self.select_case, self.__cases_box)
        middle_panel.Add(self.__cases_box, 0, wx.BOTTOM | wx.EXPAND, 5)
        middle_button_panel = wx.BoxSizer(wx.HORIZONTAL)

        self.__import_case = wx.Button(panel, label="Import case")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.import_case(), self.__import_case)
        middle_button_panel.Add(self.__import_case, 0, wx.RIGHT, 5)

        self.__delete_case = wx.Button(panel, label="Delete case")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.delete_case(), self.__delete_case)
        self.__delete_case.Enable(False)
        middle_button_panel.Add(self.__delete_case, 0, wx.RIGHT, 5)

        self.__case_filter = wx.Button(panel, label="Case filter")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.open_case_filter(), self.__case_filter)
        middle_button_panel.Add(self.__case_filter, 0, 0, 0)

        middle_panel.Add(middle_button_panel, 0, wx.BOTTOM, 5)
        middle_box = wx.StaticBox(panel, label="Case info")
        middle_panel.Add(middle_box, 1, wx.EXPAND, 0)

        middle_box_general_layout = wx.BoxSizer(wx.VERTICAL)
        middle_box_main_layout = wx.BoxSizer(wx.VERTICAL)

        short_name_caption = wx.StaticText(middle_box, label="Short name")
        middle_box_main_layout.Add(short_name_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__case_short_name_box = wx.TextCtrl(middle_box)
        middle_box_main_layout.Add(self.__case_short_name_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        long_name_caption = wx.StaticText(middle_box, label="Long name")
        middle_box_main_layout.Add(long_name_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__case_long_name_box = wx.TextCtrl(middle_box)
        middle_box_main_layout.Add(self.__case_long_name_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        stimulation_caption = wx.StaticText(middle_box, label="Stimulation")
        middle_box_main_layout.Add(stimulation_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__stimulation_box = wx.TextCtrl(middle_box)
        middle_box_main_layout.Add(self.__stimulation_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        additional_stimulation_caption = wx.StaticText(middle_box, label="Additional stimulation")
        middle_box_main_layout.Add(additional_stimulation_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__additional_stimulation_box = wx.TextCtrl(middle_box)
        middle_box_main_layout.Add(self.__additional_stimulation_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        special_conditions_caption = wx.StaticText(middle_box, label="Special conditions")
        middle_box_main_layout.Add(special_conditions_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__special_conditions_box = wx.TextCtrl(middle_box)
        middle_box_main_layout.Add(self.__special_conditions_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        additional_information_caption = wx.StaticText(middle_box, label="Additional information")
        middle_box_main_layout.Add(additional_information_caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__additional_information_box = wx.TextCtrl(middle_box, style=wx.TE_MULTILINE | wx.TE_WORDWRAP,
                                                        size=(100, 70))
        middle_box_main_layout.Add(self.__additional_information_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        self.__save_case_info = wx.Button(middle_box, label="Save case info")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.save_case_info(), self.__save_case_info)
        middle_box_main_layout.Add(self.__save_case_info, 0, 0, 0)

        middle_box_general_layout.Add(middle_box_main_layout, 1, wx.ALL | wx.EXPAND, 5)
        middle_box.SetSizer(middle_box_general_layout)

        return middle_panel

    def __create_case_study(self, panel):
        single_case_box = wx.StaticBoxSizer(wx.VERTICAL, panel, label="Case study")
        right_panel_main_layout = wx.FlexGridSizer(2, 5, 5)

        case_info_caption = wx.StaticText(panel, label="Case info")
        right_panel_main_layout.Add(case_info_caption, 1, wx.EXPAND, 0)

        self.__case_info_label = wx.StaticText(panel, label="")
        font = self.__case_info_label.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.__case_info_label.SetFont(font)
        right_panel_main_layout.Add(self.__case_info_label, 1, wx.EXPAND, 0)

        native_data_caption = wx.StaticText(panel, label="Native data")
        right_panel_main_layout.Add(native_data_caption, 1, wx.EXPAND, 0)

        self.__native_data_label = wx.StaticText(panel, label="")
        self.__native_data_label.SetFont(font)
        right_panel_main_layout.Add(self.__native_data_label, 1, wx.EXPAND, 0)

        aux_label = wx.StaticText(panel, label="")
        right_panel_main_layout.Add(aux_label, 1, wx.EXPAND, 0)

        self.__native_data_manager = wx.Button(panel, label="Open data manager")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.open_native_data_manager(), self.__native_data_manager)
        self.__native_data_manager.Enable(False)
        right_panel_main_layout.Add(self.__native_data_manager, 1, wx.EXPAND, 0)

        roi_caption = wx.StaticText(panel, label="ROI")
        right_panel_main_layout.Add(roi_caption, 0, wx.EXPAND, 0)

        self.__roi_label = wx.StaticText(panel, label="")
        self.__roi_label.SetFont(font)
        right_panel_main_layout.Add(self.__roi_label, 1, wx.EXPAND, 0)

        aux_label = wx.StaticText(panel, label="")
        right_panel_main_layout.Add(aux_label, 0, wx.EXPAND, 0)

        self.__roi_data_manager = wx.Button(panel, label="Open manager")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.open_roi_data_manager(), self.__roi_data_manager)
        self.__roi_data_manager.Enable(False)
        right_panel_main_layout.Add(self.__roi_data_manager, 0, wx.EXPAND, 0)

        trace_analysis_caption = wx.StaticText(panel, label="Trace list")
        right_panel_main_layout.Add(trace_analysis_caption, 0, wx.EXPAND, 0)

        self.__trace_analysis_label = wx.StaticText(panel, label="")
        self.__trace_analysis_label.SetFont(font)
        right_panel_main_layout.Add(self.__trace_analysis_label, 0, wx.EXPAND, 0)

        aux_label = wx.StaticText(panel, label="")
        right_panel_main_layout.Add(aux_label, 0, wx.EXPAND, 0)

        self.__trace_analysis_manager = wx.Button(panel, label="Open list")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.open_trace_analysis_manager(), self.__trace_analysis_manager)
        self.__trace_analysis_manager.Enable(False)
        right_panel_main_layout.Add(self.__trace_analysis_manager, 0, wx.EXPAND, 0)

        averaged_maps_caption = wx.StaticText(panel, label="Map list")
        right_panel_main_layout.Add(averaged_maps_caption, 0, wx.EXPAND, 0)

        self.__averaged_maps_label = wx.StaticText(panel, label="")
        self.__averaged_maps_label.SetFont(font)
        right_panel_main_layout.Add(self.__averaged_maps_label, 0, wx.EXPAND, 0)

        aux_label = wx.StaticText(panel, label="")
        right_panel_main_layout.Add(aux_label, 0, wx.EXPAND, 0)

        self.__averaged_maps_manager = wx.Button(panel, label="Open list")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.open_averaged_maps_manager(), self.__averaged_maps_manager)
        self.__averaged_maps_manager.Enable(False)
        right_panel_main_layout.Add(self.__averaged_maps_manager, 0, wx.EXPAND, 0)

        single_case_box.Add(right_panel_main_layout, 1, wx.ALL | wx.EXPAND, 5)

        self.__include_auto_box = wx.CheckBox(panel, label="In autoprocess and autocompress")
        self.Bind(wx.EVT_CHECKBOX, self.set_autoprocess, self.__include_auto_box)
        self.__include_auto_box.Enable(False)
        single_case_box.Add(self.__include_auto_box, 0, wx.LEFT | wx.RIGHT | wx.EXPAND | wx.BOTTOM, 5)

        return single_case_box

    def __create_auto_compress(self, panel):
        autocompress_box = wx.StaticBoxSizer(wx.VERTICAL, panel, label="Autocompress")
        main_layout = wx.BoxSizer(wx.HORIZONTAL)

        autocompress = wx.Button(panel, label="Compress")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.autocompress(), autocompress)
        main_layout.Add(autocompress, 0, wx.RIGHT, 5)

        autodecompress = wx.Button(panel, label="Decompress")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.autodecompress(), autodecompress)
        main_layout.Add(autodecompress, 0, 0, 0)

        autocompress_box.Add(main_layout, 0, wx.ALL | wx.EXPAND, 5)
        return autocompress_box

    def __create_auto_process(self, panel):
        autoprocess_box = wx.StaticBoxSizer(wx.VERTICAL, panel, label="Autoprocess")
        main_layout = wx.BoxSizer(wx.VERTICAL)

        self.__decompress_before_processing_box = wx.CheckBox(panel, label="Decompress before processing")
        self.Bind(wx.EVT_CHECKBOX, self.__set_decompress_before_processing, self.__decompress_before_processing_box)
        main_layout.Add(self.__decompress_before_processing_box, 0, wx.EXPAND | wx.BOTTOM, 5)

        some_actions_box = wx.BoxSizer(wx.HORIZONTAL)
        self.__extract_frame_button = wx.Button(panel, label="Extract frame")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.extract_frame(), self.__extract_frame_button)
        some_actions_box.Add(self.__extract_frame_button, 0, wx.RIGHT, 5)

        autofilter = wx.Button(panel, label="Map filtration")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.do_autofilter(), autofilter)
        some_actions_box.Add(autofilter)
        main_layout.Add(some_actions_box, 0, wx.BOTTOM, 5)

        actions_box = wx.BoxSizer(wx.HORIZONTAL)

        self.__trace_analysis_button = wx.Button(panel, label="Trace analysis")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.auto_trace_analysis(), self.__trace_analysis_button)
        actions_box.Add(self.__trace_analysis_button, 0, wx.RIGHT, 5)

        self.__auto_average_maps_box = wx.Button(panel, label="Average maps")
        self.Bind(wx.EVT_BUTTON, lambda evt: self.auto_average_maps(), self.__auto_average_maps_box)
        actions_box.Add(self.__auto_average_maps_box, 0, 0, 0)

        main_layout.Add(actions_box, 0, wx.EXPAND | wx.BOTTOM, 5)
        pandas_layout = wx.BoxSizer(wx.HORIZONTAL)

        pandas_button = wx.Button(panel, label="Result table")
        pandas_button.Bind(wx.EVT_BUTTON, lambda event: self.create_pandas_table())
        pandas_layout.Add(pandas_button, 0, wx.RIGHT)

        main_layout.Add(pandas_layout, 0, wx.EXPAND)
        autoprocess_box.Add(main_layout, 0, wx.ALL | wx.EXPAND, 5)
        return autoprocess_box

    def __create_right_panel(self, panel):
        right_panel = wx.BoxSizer(wx.VERTICAL)

        single_case_box = self.__create_case_study(panel)
        right_panel.Add(single_case_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        autocompress_box = self.__create_auto_compress(panel)
        right_panel.Add(autocompress_box, 0, wx.BOTTOM | wx.EXPAND, 5)

        autoprocess_box = self.__create_auto_process(panel)
        right_panel.Add(autoprocess_box, 0, wx.EXPAND, 0)

        service_box = wx.StaticBoxSizer(wx.VERTICAL, panel, label="Service")

        update = wx.Button(panel, label="Update")
        update.Bind(wx.EVT_BUTTON, lambda event: self.update_iman())
        service_box.Add(update, 0)

        right_panel.Add(service_box, 0, wx.EXPAND | wx.TOP, 5)
        return right_panel

    def new_animal(self):
        animal_dlg = wx.TextEntryDialog(self, "Please, enter the valid folder name", "New animal")
        if animal_dlg.ShowModal() != wx.ID_OK:
            return
        folder_name = animal_dlg.GetValue()
        if folder_name == "":
            wx.MessageDialog(self, "The folder name shall be non-empty", "Creating animal failed",
                             wx.OK | wx.ICON_ERROR | wx.CENTRE).ShowModal()
            return
        try:
            self.__animal = manifest.Animal(folder_name)
            specimen = self.__animal['specimen']
            self.__animals[specimen] = self.__animal
            self.__animals.save()
            self.load_all_animals()
        except IOError as err:
            wx.MessageDialog(self, "Failed to create the folder corresponding to an animal", "Creating animal failed",
                             wx.OK | wx.ICON_ERROR | wx.CENTRE).ShowModal()
            print(err)

    def delete_animal(self):
        try:
            specimen = self.__animal['specimen']
            self.__animal = None
            del self.__animals[specimen]
            self.__animals.save()
            self.load_all_animals()
        except IOError as err:
            wx.MessageDialog(self, "Failed to delete the animal", "Delete the animal", wx.OK | wx.ICON_ERROR). \
                ShowModal()
            print(err)

    def open_animal_filter(self):
        try:
            animal_filter = self.__animals.get_animal_filter()
            animal_filter_dlg = AnimalFilterDlg(self, animal_filter)
            if animal_filter_dlg.ShowModal() == wx.ID_CANCEL:
                return
            self.__animals.save()
        except Exception as err:
            print("Error class:", err.__class__.__name__)
            print("Error name:", err)
            dlg = wx.MessageDialog(self, str(err), "Animal filter", wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def save_animal_info(self):
        try:
            old_specimen = self.__animal['specimen']
            if self.__specimen_box.GetValue().find('_') != -1:
                dlg = wx.MessageDialog(self, "Please, don't use '_' in animal's name", "Save animal info",
                                       wx.OK | wx.CENTRE | wx.ICON_ERROR)
                dlg.ShowModal()
                return
            self.__animal['specimen'] = self.__specimen_box.GetValue()
            self.__animal['conditions'] = self.__conditions_box.GetValue()
            self.__animal['recording_site'] = self.__recording_site_box.GetValue()
            self.__animals.replace_key(old_specimen, self.__animal['specimen'])
            self.__animals.save()
            self.load_all_animals()
        except IOError as err:
            wx.MessageDialog(self, "Failed to save the data", "Save animal info", wx.OK | wx.ICON_ERROR)
            print(err)

    def import_case(self):
        import_case_dialog = ImportCaseDialog(self)
        import_mode = import_case_dialog.ShowModal()
        if import_mode == wx.ID_CANCEL:
            return
        dialog = wx.FileDialog(self, "Import cases", self.__working_dir,
                               style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)
        dialog_result = dialog.ShowModal()
        if dialog_result == wx.ID_CANCEL:
            return
        file_list = dialog.GetPaths()
        valid_files, invalid_files = sfiles.get_file_info(file_list)
        if ImportCaseManager(self, valid_files, invalid_files,
                             import_mode == ImportCaseDialog.ID_LINK,
                             self.__animal['folder_full_name']).ShowModal() == wx.ID_OK:
            self.load_cases()

    def delete_case(self):
        if self.__case['short_name'] is None:
            key = self.__case['filename']
        else:
            key = self.__case['short_name']
        self.__case = None
        del self.__cases[key]
        self.__cases.save()
        self.load_cases()

    def open_case_filter(self):
        try:
            case_filter = self.__cases.get_case_filter()
            case_filter_dlg = CaseFilterDlg(self, case_filter)
            if case_filter_dlg.ShowModal() == wx.ID_CANCEL:
                return
            self.__cases.save()
        except Exception as err:
            print("Error class:", err.__class__.__name__)
            print("Error message:", err)
            dlg = wx.MessageDialog(self, str(err), "Case Filter", wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def save_case_info(self):
        if self.__case_short_name_box.GetValue() == "":
            short_name = self.__case['filename']
        else:
            short_name = self.__case_short_name_box.GetValue()
        if short_name.find('_') != -1:
            dlg = wx.MessageDialog(self, "Please, don't use '_' in case name", "Save case info",
                                   wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()
            return
        self.__case.set_properties(
            short_name=short_name,
            long_name=self.__case_long_name_box.GetValue(),
            stimulation=self.__stimulation_box.GetValue(),
            special_conditions=self.__special_conditions_box.GetValue(),
            additional_stimulation=self.__additional_stimulation_box.GetValue(),
            additional_information=self.__additional_information_box.GetValue(),
            imported=False
        )
        self.__cases.save()
        self.load_cases()

    def open_native_data_manager(self):
        try:
            fullname = self.__animal['specimen'] + "_" + self.__case['short_name']
            manager = NativeDataManager(self, self.__case, fullname)
            manager.ShowModal()
            manager.close()
            self.__cases.save()
            self.load_cases()
        except Exception as err:
            wx.MessageDialog(self, str(err), "Native data manager",
                             style=wx.OK | wx.CENTRE | wx.ICON_ERROR).ShowModal()

    def open_roi_data_manager(self):
        fullname = "{0}_{1}".format(self.__animal['specimen'], self.__case['short_name'])
        try:
            dlg = RoiManager(self, self.__case, fullname)
            dlg.ShowModal()
            self.load_case()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), fullname, wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def open_trace_analysis_manager(self):
        try:
            dlg = TraceResultListDlg(self, self.__case)
            dlg.ShowModal()
            self.__cases.save()
            self.load_cases()
        except Exception as err:
            self.show_error_message(self, err, "Trace list")

    def open_averaged_maps_manager(self):
        try:
            dlg = MapResultListDlg(self, self.__case)
            dlg.ShowModal()
            self.__cases.save()
            self.load_cases()
        except Exception as err:
            self.show_error_message(self, err, "Map list")

    def set_autoprocess(self, evt):
        checked = evt.IsChecked()
        self.__case['auto'] = checked
        self.__cases.save()

    def save_manifest(self):
        self.__cases.save()

    def set_averaged_maps_not_found(self):
        self.__averaged_maps_label.SetLabel("Not found")
        self.__averaged_maps_label.SetForegroundColour("red")
        self.__averaged_maps_manager.Enable(False)

    def set_averaged_maps_ready(self):
        self.__averaged_maps_label.SetLabel("Ready for analysis")
        self.__averaged_maps_label.SetForegroundColour("green")
        self.__averaged_maps_manager.Enable(True)

    def set_traces_not_found(self):
        self.__trace_analysis_label.SetLabel("Not found")
        self.__trace_analysis_label.SetForegroundColour("red")
        self.__trace_analysis_manager.Enable(False)

    def set_traces_ready(self):
        self.__trace_analysis_label.SetLabel("Ready for analysis")
        self.__trace_analysis_label.SetForegroundColour("green")
        self.__trace_analysis_manager.Enable(True)

    def set_roi_not_found(self):
        self.__roi_label.SetLabel("Not found")
        self.__roi_label.SetForegroundColour("red")
        self.__roi_data_manager.Enable(True)
        self.__roi_exist = False

    def set_roi_ready(self):
        self.__roi_label.SetLabel("Ready to apply")
        self.__roi_label.SetForegroundColour("green")
        self.__roi_data_manager.Enable(True)
        self.__roi_exist = True

    def set_native_data_not_found(self):
        self.__native_data_label.SetLabel("Not found")
        self.__native_data_label.SetForegroundColour("red")
        self.__native_data_manager.Enable(False)

    def set_native_data_compressed(self):
        self.__native_data_label.SetLabel("Compressed")
        self.__native_data_label.SetForegroundColour("orange")
        self.__native_data_manager.Enable(True)
        self.__compressed_state = True

    def set_native_data_ready(self):
        self.__native_data_label.SetLabel("Ready to use")
        self.__native_data_label.SetForegroundColour("green")
        self.__native_data_manager.Enable(True)
        self.__compressed_state = False

    def set_case_info_present(self):
        self.__case_info_label.SetLabel("Present")
        self.__case_info_label.SetForegroundColour("green")

    def set_case_info_not_present(self):
        self.__case_info_label.SetLabel("Not present")
        self.__case_info_label.SetForegroundColour("red")
        self.set_native_data_not_found()
        self.set_roi_not_found()
        self.__roi_data_manager.Enable(False)
        self.set_traces_not_found()
        self.set_averaged_maps_not_found()
        self.__include_auto_box.Enable(False)

    def autocompress(self):
        try:
            autocompress_dlg = AutocompressDlg(self, self.__animals.get_animal_filter())
            if not autocompress_dlg.get_ready():
                return
            autocompress_dlg.ShowModal()
            self.load_cases()
        except Exception as err:
            self.show_error_message(self, err, "Autocompress")

    def autodecompress(self):
        try:
            autodecompress_dlg = AutodecompressDlg(self, self.__animals.get_animal_filter())
            if not autodecompress_dlg.get_ready():
                return
            autodecompress_dlg.ShowModal()
            self.load_cases()
        except Exception as err:
            self.show_error_message(self, err, "Autodecompress")

    def is_decompress_before_processing(self):
        return self.__decompress_before_processing

    def __set_decompress_before_processing(self, evt):
        self.__decompress_before_processing = evt.IsChecked()

    def extract_frame(self):
        try:
            autoextract_dlg = AutoFrameExtractDlg(self, self.__animals.get_animal_filter())
            autoextract_dlg.ShowModal()
        except Exception as err:
            self.show_error_message(self, err, "Extract frame")

    @staticmethod
    def show_error_message(parent, err, caption):
        print("Exception class:", err.__class__.__name__)
        print("Exception name:", err)
        dlg = wx.MessageDialog(parent, str(err), caption, wx.OK | wx.CENTRE | wx.ICON_ERROR)
        dlg.ShowModal()

    def auto_trace_analysis(self):
        try:
            autodecompress = self.is_decompress_before_processing()
            autotrace_dlg = AutotraceDlg(self, self.__animals.get_animal_filter(), autodecompress)
            if autotrace_dlg.get_ready():
                autotrace_dlg.ShowModal()
            self.load_cases()
        except Exception as err:
            self.show_error_message(self, err, "Auto trace")

    def auto_average_maps(self):
        try:
            autodecompress = self.is_decompress_before_processing()
            autoaverage_dlg = AutoaverageDlg(self, self.__animals.get_animal_filter(), autodecompress)
            if autoaverage_dlg.get_ready():
                autoaverage_dlg.ShowModal()
            self.load_cases()
        except Exception as err:
            self.show_error_message(self, err, "Autoaverage")

    def do_autofilter(self):
        try:
            autodecompress = self.is_decompress_before_processing()
            autofilter_dlg = AutofilterDlg(self, self.__animals.get_animal_filter(), autodecompress)
            if autofilter_dlg.get_ready():
                autofilter_dlg.ShowModal()
            self.load_cases()
        except Exception as err:
            self.show_error_message(self, err, "Autofilter")

    def open_working_dir(self, dir):
        self.__working_dir = dir
        try:
            self.__animals = manifest.Animals(dir)
            self.__animal = None
            self.load_all_animals()
        except IOError as err:
            wx.MessageDialog(self, "Failed to open the data", "Fatal error", wx.OK | wx.ICON_ERROR).ShowModal()
            raise err

    def load_all_animals(self):
        self.__animals_box.Clear()
        idx = 0
        for animal in self.__animals:
            self.__animals_box.Append(animal['specimen'])
            if self.__animal is not None and animal == self.__animal:
                self.__animals_box.SetSelection(idx)
            idx += 1
        if self.__animal is not None:
            self.load_animal()
        else:
            self.clear_animal_info()

    def select_animal(self, event):
        self.__animal = self.__animals[event.GetString()]
        self.load_animal()

    def load_animal(self):
        self.__delete_animal.Enable(True)
        self.__specimen_box.SetValue(self.__animal['specimen'])
        self.__conditions_box.SetValue(self.__animal['conditions'])
        self.__recording_site_box.SetValue(self.__animal['recording_site'])
        self.__save_animal_info.Enable(True)
        self.__import_case.Enable(True)
        self.__cases = None
        self.__case = None
        self.load_cases()

    def clear_animal_info(self):
        self.__delete_animal.Enable(False)
        self.__specimen_box.SetValue("")
        self.__conditions_box.SetValue("")
        self.__recording_site_box.SetValue("")
        self.__save_animal_info.Enable(False)
        self.__import_case.Enable(False)

    def load_cases(self):
        self.__cases = manifest.CasesList(self.__animal)
        self.__cases_box.Clear()
        idx = 0
        for case in self.__cases:
            if case['short_name'] is None:
                case_name = case['filename']
            else:
                case_name = case['short_name']
            self.__cases_box.Append(case_name)
            if self.__case is not None:
                if case['short_name'] is None:
                    if self.__case['filename'] == case['filename']:
                        self.__case = case
                        self.__cases_box.SetSelection(idx)
                else:
                    if self.__case['short_name'] == case['short_name']:
                        self.__case = case
                        self.__cases_box.SetSelection(idx)
            idx += 1
        if self.__case is not None:
            self.load_case()
        else:
            self.clear_case_info()

    def select_case(self, event):
        selection = event.GetString()
        self.__case = self.__cases[selection]
        self.load_case()

    def load_case(self):
        self.__delete_case.Enable(True)
        self.__save_case_info.Enable(True)
        if self.__case['imported']:
            self.set_case_info_not_present()
            self.__case_short_name_box.SetValue("")
            self.__case_long_name_box.SetValue("")
            self.__stimulation_box.SetValue("")
            self.__additional_stimulation_box.SetValue("")
            self.__special_conditions_box.SetValue("")
            self.__additional_information_box.SetValue("")
            self.__include_auto_box.SetValue(False)
            self.__include_auto_box.Enable(False)
        else:
            self.set_case_info_present()
            if not self.__case.native_data_files_exist():
                if not self.__case.compressed_data_files_exist():
                    self.set_native_data_not_found()
                else:
                    self.set_native_data_compressed()
            else:
                self.set_native_data_ready()
            if not self.__case.roi_exist():
                self.set_roi_not_found()
            else:
                self.set_roi_ready()
            if not self.__case.auto_traces_exist():
                self.set_traces_not_found()
            else:
                self.set_traces_ready()
            if not self.__case.averaged_maps_exist():
                self.set_averaged_maps_not_found()
            else:
                self.set_averaged_maps_ready()
            self.__case_short_name_box.SetValue(self.__case['short_name'])
            self.__case_long_name_box.SetValue(self.__case['long_name'])
            self.__stimulation_box.SetValue(self.__case['stimulation'])
            self.__additional_stimulation_box.SetValue(self.__case['additional_stimulation'])
            self.__special_conditions_box.SetValue(self.__case['special_conditions'])
            self.__additional_information_box.SetValue(self.__case['additional_information'])
            self.__include_auto_box.Enable(True)
            self.__include_auto_box.SetValue(self.__case['auto'])

    def clear_case_info(self):
        self.__cases_box.SetSelection(wx.NOT_FOUND)
        self.__delete_case.Enable(False)
        self.__save_case_info.Enable(False)
        self.__case_info_label.SetLabel("")
        self.__native_data_label.SetLabel("")
        self.__roi_label.SetLabel("")
        self.__trace_analysis_label.SetLabel("")
        self.__averaged_maps_label.SetLabel("")
        self.__include_auto_box.SetValue(False)
        self.__include_auto_box.Enable(False)
        for button in [self.__averaged_maps_manager, self.__roi_data_manager, self.__native_data_manager,
                       self.__trace_analysis_manager]:
            button.Enable(False)

    def create_pandas_table(self):
        try:
            pandas_box = PandasBox(self, self.__animals)
            pandas_box.ShowModal()
        except Exception as err:
            self.show_error_message(self, err, "Create table")

    def update_iman(self):
        try:
            print("Update iman...")
            password_dlg = wx.PasswordEntryDialog(self, "Please, enter your account password ",
                                                  "Update IMAN")
            if password_dlg.ShowModal() == wx.ID_CANCEL:
                return
            password = password_dlg.GetValue()
            pdlg = wx.ProgressDialog("Update", "Update of IMAN", maximum=3, parent=self)
            try:
                pdlg.Update(0, "Downloading new version of the package")
                self.__git()
                os.chdir("ihna_kozhuhov_image_analysis")
                pdlg.Update(1, "Building the package from sources")
                self.__build()
                pdlg.Update(2, "Installation of the package from sources")
                self.__install(password)
                os.chdir("..")
                os.system("rm -rf ihna_kozhuhov_image_analysis")
                final_dlg = wx.MessageDialog(self, "Update completed successfully. Press OK to close the program\n"
                                                   "Next, run the program again",
                                             style=wx.OK | wx.CENTRE | wx.ICON_INFORMATION)
                final_dlg.ShowModal()
                self.Close()
            except Exception as err:
                pdlg.Destroy()
                raise err
            pdlg.Destroy()
        except Exception as err:
            self.show_error_message(self, err, "Update IMAN")

    def __git(self):
        if os.system("git clone https://github.com/serik1987/ihna_kozhuhov_image_analysis") != 0:
            raise RuntimeError("Failure to download the package. Install the package manually")

    def __build(self):
        if os.system("python3 setup.py build") != 0:
            raise RuntimeError("Failure to build the package. Install the package manually")

    def __install(self, password):
        command = "echo %s | sudo -k -S python3 setup.py install" % password
        if os.system(command) != 0:
            raise RuntimeError("Failure to install the package. May be, you entered incorrect password")
