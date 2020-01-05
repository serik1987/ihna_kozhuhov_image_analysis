# -*- coding: utf-8


import wx
from ihna.kozhukhov.imageanalysis.manifest import SimpleRoi


class ManualRoiSelectDlg(wx.Dialog):
    """
    Represents a dialog for manual ROI select

    Usage:
    dlg = ManualRoiSelectDlg(parent_window, data_name)
    if dlg.ShowModal() == wx.ID_CANCEL:
        return
    the_roi = dlg.get_roi()
    """

    __name_box = None
    __left_box = None
    __right_box = None
    __top_box = None
    __bottom_box = None
    __roi = None

    def __init__(self, parent, fullname):
        super().__init__(parent, title="Manual ROI select: " + fullname, size=(500, 500))
        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)

        self.__name_box = self.__create_box(main_panel, main_layout, "ROI name (*)")
        self.__left_box = self.__create_box(main_panel, main_layout, "Left border, px (*)")
        self.__right_box = self.__create_box(main_panel, main_layout, "Right border, px (*)")
        self.__top_box = self.__create_box(main_panel, main_layout, "Top border, px (*)")
        self.__bottom_box = self.__create_box(main_panel, main_layout, "Bottom border, px (*)")

        buttons_panel = wx.BoxSizer(wx.HORIZONTAL)

        ok_button = wx.Button(main_panel, label="OK")
        self.Bind(wx.EVT_BUTTON, lambda event: self.finalize_roi_selection(), ok_button)
        buttons_panel.Add(ok_button, 0, wx.RIGHT, 20)

        cancel_button = wx.Button(main_panel, label="Cancel")
        self.Bind(wx.EVT_BUTTON, lambda event: self.Close(), cancel_button)
        buttons_panel.Add(cancel_button)

        main_layout.Add(buttons_panel, 0, wx.ALIGN_CENTER | wx.TOP, 10)

        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizerAndFit(general_layout)
        self.Centre()
        self.Fit()

    def __create_box(self, parent, sizer, name):
        caption = wx.StaticText(parent, label=name)
        sizer.Add(caption, 0, wx.BOTTOM | wx.EXPAND, 5)

        box = wx.TextCtrl(parent)
        sizer.Add(box, 0, wx.BOTTOM | wx.EXPAND, 5)

        return box

    def finalize_roi_selection(self):
        try:
            roi = SimpleRoi()
            roi_name = self.__name_box.GetValue()
            if roi_name == "":
                raise ValueError("ROI name shall not be empty")
            roi.set_name(roi_name)
            try:
                left_border = int(self.__left_box.GetValue())
            except ValueError:
                raise ValueError("Please, check the validity of the ROI left border")
            try:
                right_border = int(self.__right_box.GetValue())
            except ValueError:
                raise ValueError("Please, check the validity of the ROI right border")
            try:
                top_border = int(self.__top_box.GetValue())
            except ValueError:
                raise ValueError("Please, check the validity of the ROI top border")
            try:
                bottom_border = int(self.__bottom_box.GetValue())
            except ValueError:
                ValueError("Please, check the validity of the ROI bottom border")
            roi.set_left(left_border)
            roi.set_right(right_border)
            roi.set_top(top_border)
            roi.set_bottom(bottom_border)
            self.__roi = roi
            self.EndModal(wx.OK)
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), "Manual ROI definition")
            dlg.ShowModal()

    def get_roi(self):
        return self.__roi
