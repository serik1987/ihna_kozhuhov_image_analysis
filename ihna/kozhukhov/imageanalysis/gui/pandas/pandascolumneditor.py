# -*- coding: utf-8

import wx


class PandasColumnEditor(wx.Panel):

    _column = None

    def __init__(self, parent, column=None):
        self._column = column
        super().__init__(parent)

    def additional_check(self):
        raise NotImplementedError("PandasColumnEditor.additional_check")

    def _get_column_class(self):
        raise NotImplementedError("PandasColumnEditor._get_column_class")

    def _get_column_properties(self):
        raise NotImplementedError("PandasColumnEditor._get_column_properties")

    def get_column(self, column_name):
        self._column = self._get_column_class()()
        self._get_column_properties()
        self._column.set_name(column_name)
        return self._column

    def set_column(self):
        raise NotImplementedError("PandasColumnEditor.set_column")
