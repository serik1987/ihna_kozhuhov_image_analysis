# -*- coding: utf-8

import wx
from .datatonumberprocessor import DataToNumberProcessor
from .twomapselector import TwoMapSelector


class TwoDataToNumberProcessor(DataToNumberProcessor):

    _second_map = None

    def _check_input_data(self):
        selector = TwoMapSelector(self._parent, self._considering_case)
        if selector.ShowModal() == wx.ID_CANCEL:
            self._input_data = None
            self._second_map = None
            return
        self._input_data = selector.get_first_map()
        self._second_map = selector.get_second_map()
        self._input_data.load_data()
        self._second_map.load_data()
        self._check_two_maps()

    def _check_two_maps(self):
        raise NotImplementedError("Check two maps")

    def _process(self):
        self._process_maps()
        temp = self._input_data
        self._input_data = self._second_map
        self._second_map = temp

    def _process_maps(self):
        raise NotImplementedError("_process")
