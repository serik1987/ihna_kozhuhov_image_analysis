# -*- coding: utf-8

from scipy.signal import butter, buttord
from .filterbox import FilterBox


class ButterBox(FilterBox):

    _filter_properties = ["broadband", "manual"]

    def _get_filter_name(self):
        return "butter"

    def get_coefficients(self):
        N = self.get_order()
        Wn = self._get_std_passband()
        btype = self._get_btype()
        b, a = butter(N, Wn, btype)
        return b, a
