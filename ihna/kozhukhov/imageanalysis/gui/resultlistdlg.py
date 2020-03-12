# -*- coding: utf-8

import wx


class ResultListDlg(wx.Dialog):

    __case = None

    def __init__(self, parent, case):
        self.__case = case
        super().__init__(self, parent, title="Sample title", size=(700, 500))

        self.Centre()

    def get_case(self):
        return self.__case
