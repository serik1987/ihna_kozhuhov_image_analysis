# -*- coding: utf-8

import wx
from .pandasfeaturecolumn import PandasFeatureColumn
from .pandascolumneditor import PandasColumnEditor


class PandasFeatureColumnEditor(PandasColumnEditor):

    __prefix_name_box = None
    __prefix_name_included_box = None
    __postfix_name_box = None
    __postfix_name_included_box = None
    __result_minor_name_box = None
    __feature_name_box = None

    def __init__(self, parent, column):
        super().__init__(parent, column)
        main_layout = wx.GridBagSizer(vgap=5, hgap=5)

        prefix_name_caption = wx.StaticText(self, label="Data name prefix")
        main_layout.Add(prefix_name_caption, pos=(0, 0), flag=wx.ALIGN_CENTER_VERTICAL)

        self.__prefix_name_box = wx.TextCtrl(self)
        main_layout.Add(self.__prefix_name_box, pos=(0, 1), flag=wx.EXPAND)

        self.__prefix_name_included_box = wx.CheckBox(self, label="Include all")
        self.__prefix_name_included_box.Bind(wx.EVT_CHECKBOX, lambda event: self.__check_include_all_prefixes())
        main_layout.Add(self.__prefix_name_included_box, pos=(0, 2), flag=wx.ALIGN_CENTER_VERTICAL)

        postfix_name_caption = wx.StaticText(self, label="Data name postfix")
        main_layout.Add(postfix_name_caption, pos=(1, 0), flag=wx.ALIGN_CENTER_VERTICAL)

        self.__postfix_name_box = wx.TextCtrl(self)
        main_layout.Add(self.__postfix_name_box, pos=(1, 1), flag=wx.EXPAND)

        self.__postfix_name_included_box = wx.CheckBox(self, label="Include all")
        self.__postfix_name_included_box.Bind(wx.EVT_CHECKBOX, lambda event: self.__check_include_all_postfixes())
        main_layout.Add(self.__postfix_name_included_box, pos=(1, 2), flag=wx.ALIGN_CENTER_VERTICAL)

        minor_name_caption = wx.StaticText(self, label="Data minor name")
        main_layout.Add(minor_name_caption, pos=(2, 0), flag=wx.ALIGN_CENTER_VERTICAL)

        self.__result_minor_name_box = wx.TextCtrl(self)
        main_layout.Add(self.__result_minor_name_box, pos=(2, 1), flag=wx.EXPAND, span=(1, 2))

        feature_name_caption = wx.StaticText(self, label="Feature name")
        main_layout.Add(feature_name_caption, pos=(3, 0), flag=wx.ALIGN_CENTER_VERTICAL)

        self.__feature_name_box = wx.TextCtrl(self)
        main_layout.Add(self.__feature_name_box, pos=(3, 1), span=(1, 2), flag=wx.EXPAND)

        main_layout.AddGrowableCol(1, 1)
        self.SetSizer(main_layout)

        self.__check_include_all_postfixes()
        self.__check_include_all_prefixes()

    def __check_include_all_prefixes(self):
        if self.__prefix_name_included_box.IsChecked():
            self.__prefix_name_box.SetValue("")
            self.__prefix_name_box.Enable(False)
        else:
            self.__prefix_name_box.Enable(True)

    def __check_include_all_postfixes(self):
        if self.__postfix_name_included_box.IsChecked():
            self.__postfix_name_box.SetValue("")
            self.__postfix_name_box.Enable(False)
        else:
            self.__postfix_name_box.Enable(True)

    def additional_check(self):
        if self.__result_minor_name_box.GetValue() == "":
            raise ValueError("Please, specify the minor name")
        if self.__feature_name_box.GetValue() == "":
            raise ValueError("Please, specify the feature name")

    def _get_column_class(self):
        return PandasFeatureColumn

    def _get_column_properties(self):
        if not self.__prefix_name_included_box.IsChecked():
            self._column.set_prefix_name(self.__prefix_name_box.GetValue())
        else:
            self._column.clear_prefix_name()
        if not self.__postfix_name_included_box.IsChecked():
            self._column.set_postfix_name(self.__postfix_name_box.GetValue())
        else:
            self._column.clear_postfix_name()
        self._column.set_minor_name(self.__result_minor_name_box.GetValue())
        self._column.set_feature_name(self.__feature_name_box.GetValue())

    def set_column(self):
        prefix_name = self._column.get_prefix_name()
        if prefix_name is None:
            self.__prefix_name_included_box.SetValue(True)
            self.__prefix_name_box.SetValue("")
        else:
            self.__prefix_name_included_box.SetValue(False)
            self.__prefix_name_box.SetValue(prefix_name)
        self.__check_include_all_prefixes()

        postfix_name = self._column.get_postfix_name()
        if postfix_name is None:
            self.__postfix_name_included_box.SetValue(True)
            self.__postfix_name_box.SetValue("")
        else:
            self.__postfix_name_included_box.SetValue(False)
            self.__postfix_name_box.SetValue(postfix_name)
        self.__check_include_all_postfixes()

        self.__result_minor_name_box.SetValue(self._column.get_minor_name())
        self.__feature_name_box.SetValue(self._column.get_feature_name())
