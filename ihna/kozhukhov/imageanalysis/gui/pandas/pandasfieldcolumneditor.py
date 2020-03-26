# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.manifest import Animal, Case
from .pandasfieldcolumn import PandasFieldColumn
from .pandascolumneditor import PandasColumnEditor


class PandasFieldColumnEditor(PandasColumnEditor):

    __column_box = None

    __field_list = [
        {
            "class": Animal,
            "name": "specimen",
            "description": "specimen"
        }, {
            "class": Animal,
            "name": "conditions",
            "description": "condition"
        }, {
            "class": Animal,
            "name": "recording_site",
            "description": "recording site"
        }, {
            "class": Case,
            "name": "short_name",
            "description": "short name"
        }, {
            "class": Case,
            "name": "long_name",
            "description": "long name"
        }, {
            "class": Case,
            "name": "stimulation",
            "description": "stimulation"
        }, {
            "class": Case,
            "name": "additional_stimulation",
            "description": "additional stimulation"
        }, {
            "class": Case,
            "name": "special_conditions",
            "description": "special conditions"
        }, {
            "class": Case,
            "name": "additional_information",
            "description": "additional information"
        }
    ]

    def __init__(self, parent, column):
        super().__init__(parent, column)
        column_sizer = wx.BoxSizer(wx.HORIZONTAL)

        column_caption = wx.StaticText(self, label="Field name")
        column_sizer.Add(column_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

        column_names = ["%s's %s" % (info['class'].__name__, info['description']) for info in self.__field_list]
        self.__column_box = wx.Choice(self, choices=column_names)
        self.__column_box.SetSelection(0)
        column_sizer.Add(self.__column_box, 1, wx.EXPAND)

        self.SetSizer(column_sizer)

    def additional_check(self):
        pass

    def _get_column_class(self):
        return PandasFieldColumn

    def _get_column_properties(self):
        idx = self.__column_box.GetSelection()
        field_class = self.__field_list[idx]['class']
        field_name = self.__field_list[idx]['name']
        self._column.set_field_class(field_class)
        self._column.set_field_key(field_name)

    def set_column(self):
        idx = 0
        for field_info in self.__field_list:
            if field_info['class'] == self._column.get_field_class() \
                    and field_info['name'] == self._column.get_field_key():
                break
            idx += 1
        self.__column_box.SetSelection(idx)
