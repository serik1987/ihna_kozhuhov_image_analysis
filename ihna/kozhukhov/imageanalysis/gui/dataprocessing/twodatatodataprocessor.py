# -*- coding: utf-8

import wx
from .datatodataprocessor import DataToDataProcessor
from .twomapselector import TwoMapSelector


class TwoDataToDataProcessor(DataToDataProcessor):

    _second_map = None

    def _check_input_data(self):
        dlg = TwoMapSelector(self._parent, self._considering_case)
        if dlg.ShowModal() == wx.ID_CANCEL:
            self._input_data = None
            self._second_map = None
            return
        self._input_data = dlg.get_first_map()
        self._second_map = dlg.get_second_map()
        self._input_data.load_data()
        self._second_map.load_data()
        self._check_two_maps()

    def _check_two_maps(self):
        """
        Raises an exception if the processor is not available for such pair of maps
        """
        raise NotImplementedError("_check_two_maps")
