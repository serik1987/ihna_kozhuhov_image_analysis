# -*- coding: utf-8


import wx
from wx.lib.scrolledpanel import ScrolledPanel
from ihna.kozhukhov.imageanalysis.manifest import ComplexRoi


class DefineComplexRoiDlg(wx.Dialog):
    """
    This is dialog that is used to define the complex ROI
    """

    __roi_name_box = None
    __roi_list_box = None
    __content = None
    __roi_names = None
    __roi_list = None
    __roi = None

    def __init__(self, parent, fullname, roi_list):
        """
        Initialization

        Arguments:
            parent - the parent dialog/frame
            fullname - name to be printed at the title
            roi_list - instance of ihna.kozhukhov.imageanalysis.manifest.RoiList

        After initialization, ShowModal() it (shall return wx.ID_OK on success and then
        use get_roi() to receive the complex ROI (this is your responsibility to add the
        complex roi defined by the user to the ROI list)
        """
        super().__init__(parent, title="Define complex ROI for " + fullname, size=(300, 400))
        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)

        roi_name_box = self.__create_roi_name_box(main_panel)
        main_layout.Add(roi_name_box, 0, wx.BOTTOM | wx.EXPAND, 10)

        roi_list_box = self.__create_roi_list_box(main_panel, roi_list)
        main_layout.Add(roi_list_box, 1, wx.BOTTOM | wx.EXPAND, 10)

        buttons_box = self.__create_buttons_panel(main_panel)
        main_layout.Add(buttons_box, 0, wx.ALIGN_CENTER)

        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizer(general_layout)
        self.Centre()

        self.__roi_list = roi_list

    def __create_roi_name_box(self, parent):
        box = wx.BoxSizer(wx.VERTICAL)

        caption = wx.StaticText(parent, label="ROI name (*)")
        box.Add(caption, 0, wx.EXPAND | wx.BOTTOM, 5)

        self.__roi_name_box = wx.TextCtrl(parent)
        box.Add(self.__roi_name_box, 0, wx.EXPAND)

        return box

    def __create_roi_list_box(self, parent, roi_list):
        box = ScrolledPanel(parent)
        box.SetupScrolling(scroll_x=False, scroll_y=True, scrollToTop=False)
        subbox = wx.BoxSizer(wx.VERTICAL)
        self.__content = []

        self.__roi_names = []
        idx = 0
        for roi in roi_list:
            roi_box = wx.CheckBox(box, label=roi.get_name(), id=idx)
            self.Bind(wx.EVT_CHECKBOX, self.change_content, roi_box)
            subbox.Add(roi_box, 0, wx.EXPAND | wx.ALL, 5)
            self.__roi_names.append(roi.get_name())
            idx += 1

        box.SetSizer(subbox)
        return box

    def __create_buttons_panel(self, parent):
        box = wx.BoxSizer(wx.HORIZONTAL)

        btn_ok = wx.Button(parent, label="OK")
        self.Bind(wx.EVT_BUTTON, lambda event: self.finalize_complex_roi(), btn_ok)
        box.Add(btn_ok, wx.BOTTOM, 5)

        btn_cancel = wx.Button(parent, label="Cancel")
        self.Bind(wx.EVT_BUTTON, lambda event: self.Close(), btn_cancel)
        box.Add(btn_cancel)

        return box

    def finalize_complex_roi(self):
        name = self.__roi_name_box.GetValue()
        if name == "":
            dlg = wx.MessageDialog(self, "Please, specify name of the complex ROI", "Define complex ROI",
                                wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()
            return
        if len(self.__content) == 0:
            dlg = wx.MessageDialog(self, "Please, check all necessary simple ROI", "Define complex ROI",
                                wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()
            return

        self.__roi = ComplexRoi(self.__roi_list, name, self.__content)
        self.EndModal(wx.ID_OK)

    def change_content(self, evt):
        id = evt.GetEventObject().GetId()
        name = self.__roi_names[id]
        if evt.IsChecked():
            self.__content.append(name)
        else:
            self.__content.remove(name)

    def get_roi(self):
        return self.__roi
