# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.tracereading import TraceReaderAndCleaner as TraceReader
from .synchronization.selector import SynchronizationSelector
from .isolines.selector import IsolineSelector


class TraceAnalysisPropertiesDlg(wx.Dialog):
    """
    Provides a convenient dialog for setting trace analysis properties
    """

    __train = None
    __parent = None
    __sync_selector = None
    __isoline_selector = None
    __channel_selector = None
    __sync_signal_widgets = None
    __roi_box = None
    __roi_list = None
    __correctness_check_box = None

    def __init__(self, parent, train, roi_list):
        super().__init__(parent, title="Trace analysis properties", size=(800, 600))
        self.__parent = parent
        self.__train = train
        self.__roi_list = roi_list

        if len(roi_list) == 0:
            raise ValueError("In order to use this function you shall specify at least one ROI")

        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)

        upper_panel = self.__init_upper_panel(main_panel)
        main_layout.Add(upper_panel, 0, wx.BOTTOM, 10)

        middle_panel = self.__init_middle_panel(main_panel)
        main_layout.Add(middle_panel, 0, wx.BOTTOM, 10)

        lower_panel = self.__init_lower_panel(main_panel)
        main_layout.Add(lower_panel, 0, wx.ALIGN_CENTER)

        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizerAndFit(general_layout)
        self.Centre()
        self.Fit()

    def __init_upper_panel(self, parent):
        upper_panel = wx.BoxSizer(wx.HORIZONTAL)

        self.__sync_selector = SynchronizationSelector(parent, self.__train)
        upper_panel.Add(self.__sync_selector, 0, wx.RIGHT, 10)

        upper_right_panel = wx.BoxSizer(wx.VERTICAL)

        self.__isoline_selector = IsolineSelector(parent, self.__train)
        upper_right_panel.Add(self.__isoline_selector, 0, wx.BOTTOM | wx.EXPAND, 10)

        self.__channel_selector = self.__init_channel_selector(parent)
        upper_right_panel.Add(self.__channel_selector, 0, wx.EXPAND)

        upper_panel.Add(upper_right_panel, 0)
        return upper_panel

    def __init_channel_selector(self, parent):
        box = wx.StaticBoxSizer(wx.VERTICAL, parent, label="Trace reading")

        layout = wx.BoxSizer(wx.VERTICAL)

        self.__sync_signal_widgets = []
        try:
            if not self.__train.is_opened:
                raise RuntimeError("Please, open the train before creating this dialog box")
            for chan in range(self.__train.synchronization_channel_number):
                wid = wx.CheckBox(parent, label="Include synchronization signal # " + str(chan))
                self.__sync_signal_widgets.append(wid)
                layout.Add(wid, 0, wx.EXPAND | wx.BOTTOM, 5)

            roi_layout = wx.BoxSizer(wx.HORIZONTAL)

            roi_caption = wx.StaticText(parent, label="ROI")
            roi_layout.Add(roi_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)

            choices = []
            for roi in self.__roi_list:
                choices.append(roi.get_name())
            self.__roi_box = wx.Choice(parent, choices=choices)
            if len(choices) != 0:
                self.__roi_box.SetSelection(0)
            roi_layout.Add(self.__roi_box, 1, wx.EXPAND)

            layout.Add(roi_layout, 0, wx.EXPAND)
        except Exception as err:
            print("Synchronization signal can't be included")
            print("Reason:", err.__class__.__name__)
            print("Comment:", str(err))

        box.Add(layout, 1, wx.EXPAND | wx.ALL, 5)
        return box

    def __init_middle_panel(self, parent):
        middle_panel = wx.BoxSizer(wx.HORIZONTAL)

        check_button = wx.Button(parent, label="Check parameters for correctness")
        check_button.Bind(wx.EVT_BUTTON, lambda evt: self.correctness_check())
        middle_panel.Add(check_button, 0, wx.RIGHT, 5)

        self.__correctness_check_box = wx.StaticText(parent,
                                                     label="Please the button at the left to check and continue")
        middle_panel.Add(self.__correctness_check_box, 0, wx.ALIGN_CENTER_VERTICAL)

        return middle_panel

    def __init_lower_panel(self, parent):
        lower_panel = wx.BoxSizer(wx.HORIZONTAL)

        ok = wx.Button(parent, label="OK")
        self.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_OK), ok)
        lower_panel.Add(ok, 0, wx.RIGHT, 5)

        cancel = wx.Button(parent, label="Cancel")
        self.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_CANCEL), cancel)
        lower_panel.Add(cancel)

        return lower_panel

    def close(self):
        """
        Deletes the file train that is associated with the dialog
        """
        del self.__train
        self.__train = None
        del self.__parent
        self.__parent = None

        self.__sync_selector.close()
        self.__isoline_selector.close()

    def correctness_check(self):
        print("PY correctness check")
        sync = self.create_synchronization()
        isoline = self.create_isoline(sync)
        print(sync)
        print(isoline)

    def get_pixel_list(self):
        roi_number = self.__roi_box.GetSelection()
        roi_name = self.__roi_box.GetItems()[roi_number]
        roi = self.__roi_list[roi_name]
        pixel_list = roi.get_coordinate_list()
        chan = 0
        for sync_signal_box in self.__sync_signal_widgets:
            if sync_signal_box.IsChecked():
                pixel_list.append(('SYNC', chan))
            chan += 1
        pixel_list.append(('TIME', 0))
        return pixel_list

    def create_synchronization(self):
        return self.__sync_selector.create_synchronization()

    def create_isoline(self, sync):
        return self.__isoline_selector.create_isoline(sync)

    def create_reader(self):
        """
        Returns a three-item tuple containing the reader, the isoline and the synchronization
        """
        if len(self.get_pixel_list()) > 2000:
            raise RuntimeError("ROI with area higher than 2000 px can't be processed without autoaverage\n"
                               "Please, set the autoaverage or select smaller ROI")
        reader = TraceReader(self.__train)
        sync = self.create_synchronization()
        isoline = self.create_isoline(sync)
        reader.isoline_remover = isoline
        reader.add_pixels(self.get_pixel_list())
        return reader, isoline, sync

    def get_roi_name(self):
        roi_number = self.__roi_box.GetSelection()
        roi_name = self.__roi_box.GetItems()[roi_number]
        return roi_name
