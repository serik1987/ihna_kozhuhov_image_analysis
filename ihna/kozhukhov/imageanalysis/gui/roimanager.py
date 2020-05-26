# -*- coding: utf-8

import os
import xml.etree.ElementTree as ET
import numpy as np
import wx
from wx.grid import Grid
from ihna.kozhukhov.imageanalysis.manifest import SimpleRoi
import ihna.kozhukhov.imageanalysis.sourcefiles as sfiles
from .definesimpleroidlg import DefineSimpleRoiDlg
from .definecomplexroidlg import DefineComplexRoiDlg
from .manualroiselectdlg import ManualRoiSelectDlg
from .showroidlg import ShowRoiDlg


class RoiManager(wx.Dialog):
    """
    Represents a graphical tool to add, draw and delete different ROIs
    """

    __data = None
    __fullname = None
    __table = None
    __parent = None

    __btn_import = None
    __btn_simple = None
    __btn_complex = None
    __btn_delete = None
    __btn_manual = None
    __btn_export = None
    __btn_show = None

    def __init__(self, parent, data, fullname):
        self.__parent = parent
        self.__data = data
        self.__fullname = fullname
        super().__init__(parent, title="ROI manager: " + fullname, size=(800, 600))
        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)

        self.__create_table(main_panel)
        main_layout.Add(self.__table, 1, wx.BOTTOM | wx.EXPAND, 25)
        buttons = self.__create_buttons(main_panel)
        main_layout.Add(buttons, 0, wx.BOTTOM | wx.ALIGN_CENTER, 25)

        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizer(general_layout)
        self.Centre()
        self.update_roi()

    def __create_table(self, parent):
        self.__table = Grid(parent)
        self.__table.CreateGrid(0, 6, wx.grid.Grid.wxGridSelectRows)
        self.__table.SetColLabelValue(0, "type")
        self.__table.SetColLabelValue(1, "left")
        self.__table.SetColLabelValue(2, "right")
        self.__table.SetColLabelValue(3, "top")
        self.__table.SetColLabelValue(4, "bottom")
        self.__table.SetColLabelValue(5, "area")
        self.__table.EnableEditing(False)

    def __create_buttons(self, parent):
        all_buttons = wx.BoxSizer(wx.VERTICAL)
        buttons = wx.BoxSizer(wx.HORIZONTAL)

        self.__btn_import = wx.Button(parent, label="Import ROI")
        self.Bind(wx.EVT_BUTTON, lambda event: self.import_roi(), self.__btn_import)
        buttons.Add(self.__btn_import, 0, wx.RIGHT, 5)

        self.__btn_simple = wx.Button(parent, label="Define simple ROI")
        self.Bind(wx.EVT_BUTTON, lambda event: self.define_simple_roi(), self.__btn_simple)
        buttons.Add(self.__btn_simple, 0, wx.RIGHT, 5)

        self.__btn_complex = wx.Button(parent, label="Define complex ROI")
        self.Bind(wx.EVT_BUTTON, lambda event: self.define_complex_roi(), self.__btn_complex)
        buttons.Add(self.__btn_complex, 0, wx.RIGHT, 5)

        self.__btn_delete = wx.Button(parent, label="Delete ROI")
        self.Bind(wx.EVT_BUTTON, lambda event: self.delete_roi(), self.__btn_delete)
        buttons.Add(self.__btn_delete, 0, wx.RIGHT, 5)

        self.__btn_manual = wx.Button(parent, label="Define ROI manually")
        self.Bind(wx.EVT_BUTTON, lambda event: self.manual_roi_select(), self.__btn_manual)
        buttons.Add(self.__btn_manual, 0, wx.RIGHT, 5)

        self.__btn_export = wx.Button(parent, label="Export to TXT")
        self.Bind(wx.EVT_BUTTON, lambda event: self.export_to_txt(), self.__btn_export)
        buttons.Add(self.__btn_export, 0, wx.RIGHT, 0)

        all_buttons.Add(buttons, 0, wx.BOTTOM | wx.ALIGN_CENTER, 5)
        add_buttons = wx.BoxSizer(wx.HORIZONTAL)

        self.__btn_show = wx.Button(parent, label="Show ROI on map")
        self.Bind(wx.EVT_BUTTON, lambda event: self.show_roi(), self.__btn_show)
        add_buttons.Add(self.__btn_show, 0, wx.RIGHT, 0)

        all_buttons.Add(add_buttons, 0, wx.ALIGN_CENTER)
        return all_buttons

    def update_roi(self):
        roi_list = self.__data['roi']
        old_rows = self.__table.GetNumberRows()
        if old_rows > 0:
            self.__table.DeleteRows(0, old_rows)
        if len(roi_list) > 0:
            self.__table.AppendRows(len(roi_list))
            idx = 0
            for roi in roi_list:
                self.__table.SetRowLabelValue(idx, roi.get_name())
                self.__table.SetCellValue(idx, 0, str(roi.get_type()))
                self.__table.SetCellValue(idx, 1, str(roi.get_left()))
                self.__table.SetCellValue(idx, 2, str(roi.get_right()))
                self.__table.SetCellValue(idx, 3, str(roi.get_top()))
                self.__table.SetCellValue(idx, 4, str(roi.get_bottom()))
                self.__table.SetCellValue(idx, 5, str(roi.get_area()))
                idx += 1
            for btn in (self.__btn_delete, self.__btn_show):
                btn.Enable(True)
        else:
            for btn in (self.__btn_delete, self.__btn_show):
                btn.Enable(False)
        self.__parent.save_manifest()

    def import_roi(self):
        try:
            file_dialog = wx.FileDialog(self, "Import ROI from imaging package",
                                        style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
                                        wildcard="*.xml")
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            filename = file_dialog.GetPath()
            tree = ET.parse(filename)
            data_element = tree.getroot()
            if data_element.tag != "data":
                raise ValueError("This is not an old ROI file")
            roi_element = data_element.find("map-roi")
            if roi_element is None:
                raise ValueError("This is not an old ROI file")
            roi = SimpleRoi()
            roi.set_left(roi_element.attrib['left'])
            roi.set_right(roi_element.attrib['right'])
            roi.set_top(roi_element.attrib['top'])
            roi.set_bottom(roi_element.attrib['bottom'])
            name_dialog = wx.TextEntryDialog(self, "Please, enter the ROI name", "Import ROI from imaging package")
            if name_dialog.ShowModal() == wx.ID_CANCEL:
                return
            roi.set_name(name_dialog.GetValue())
            self.__data['roi'].add(roi)
            self.update_roi()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "ROI manager", wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def get_vessel_map(self):
        pathname = self.__data['pathname']
        filename = None
        TheTrain = None
        if self.__data.native_data_files_exist():
            filename = self.__data['native_data_files'][0]
            TheTrain = sfiles.StreamFileTrain
        if self.__data.compressed_data_files_exist():
            filename = self.__data['compressed_data_files'][0]
            TheTrain = sfiles.CompressedFileTrain
        if TheTrain is None:
            return None
        fullname = os.path.join(pathname, filename)
        train = TheTrain(fullname)
        train.open()
        frame_data = train[0].body
        del train
        return frame_data

    def get_f_maps(self):
        amap = None
        pmap = None
        harmonic = 1.0
        for avg_map in self.__data.data():
            features = avg_map.get_features()
            if 'is_main' in features and features['is_main'] == 'yes':
                avg_map.load_data()
                cmap = avg_map.get_data()
                harmonic = avg_map.get_harmonic()
                amap = np.abs(cmap)
                pmap = np.angle(cmap) / harmonic
        return amap, pmap, harmonic

    def define_simple_roi(self):
        try:
            vessel_map = self.get_vessel_map()
            amap, pmap, harm = self.get_f_maps()
            dlg = DefineSimpleRoiDlg(self, self.__fullname, vessel_map, amap, pmap, harm)
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            roi = dlg.get_roi()
            self.__data['roi'].add(roi)
            self.update_roi()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Define simple ROI", wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def define_complex_roi(self):
        try:
            dlg = DefineComplexRoiDlg(self, self.__fullname, self.__data['roi'])
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            self.__data['roi'].add(dlg.get_roi())
            self.update_roi()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Define complex ROI", wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def get_selected_roi(self):
        names = []
        rows = self.__table.GetSelectedRows()
        for row in rows:
            names.append(self.__table.GetRowLabelValue(row))
        return names

    def delete_roi(self):
        names = self.get_selected_roi()
        if len(names) != 1:
            raise ValueError("Please, select a ROI to delete")
        del self.__data['roi'][names[0]]
        self.update_roi()

    def manual_roi_select(self):
        try:
            dlg = ManualRoiSelectDlg(self, self.__fullname)
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            self.__data['roi'].add(dlg.get_roi())
            self.update_roi()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Manual ROI select", wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def export_to_txt(self):
        try:
            dlg = wx.FileDialog(self, "Save ROI list to TXT", wildcard="*.txt",
                                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            S = str(self.__data['roi'])
            filename = dlg.GetPath()
            if not filename.endswith(".txt"):
                filename += ".txt"
            f = open(filename, 'w')
            f.write(S)
            f.close()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Export to TXT", wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def show_roi(self):
        try:
            names = self.get_selected_roi()
            if len(names) != 1:
                raise ValueError("Please, select a ROI to show")
            roi = self.__data['roi'][names[0]]
            vessel_map = self.get_vessel_map()
            amplitude_map, phase_map, h = self.get_f_maps()
            dlg = ShowRoiDlg(self, self.__fullname, roi, vessel_map, amplitude_map, phase_map, h)
            dlg.ShowModal()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Show ROI", wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()
