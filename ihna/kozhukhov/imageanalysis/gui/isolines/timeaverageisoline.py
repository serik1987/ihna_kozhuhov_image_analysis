# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.sourcefiles import StreamFileTrain
from ihna.kozhukhov.imageanalysis.isolines import TimeAverageIsoline
from .isoline import IsolineEditor


class TimeAverageIsolineEditor(IsolineEditor):
    """
    Represents widgets to select the TimeAverageIsoline and set all its properties
    """

    __cycles_caption = None
    __cycles_box = None
    __average_result = None

    def get_name(self) -> str:
        return "Time average"

    def __init__(self, parent, train: StreamFileTrain, selector):
        super().__init__(parent, train, selector)

    def has_properties(self):
        return True

    def enable(self):
        self.__cycles_caption.Enable(True)
        self.__cycles_box.Enable(True)

    def disable(self):
        self.__cycles_caption.Enable(False)
        self.__cycles_box.Enable(False)
        self.__average_result.SetLabel("")

    def put_properties(self):
        properties = wx.BoxSizer(wx.VERTICAL)
        cycle_layout = wx.BoxSizer(wx.HORIZONTAL)

        self.__cycles_caption = wx.StaticText(self._parent, label="Average radius, cycles")
        cycle_layout.Add(self.__cycles_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        self.__cycles_box = wx.TextCtrl(self._parent, value="1")
        cycle_layout.Add(self.__cycles_box)

        properties.Add(cycle_layout, 0, wx.BOTTOM, 5)

        self.__average_result = wx.StaticText(self._parent, label="")
        properties.Add(self.__average_result, 0, wx.EXPAND)

        return properties

    def create_isoline(self, train, sync):
        isoline = TimeAverageIsoline(train, sync)
        try:
            isoline.average_cycles = int(self.__cycles_box.GetValue())
        except ValueError:
            raise ValueError("Please, specify a correct value in the 'Average radius' box")
        return isoline

    def get_options(self):
        return "time average: {0} cycles".format(self.__cycles_box.GetValue())
