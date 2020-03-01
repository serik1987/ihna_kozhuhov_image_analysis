# -*- coding: utf-8

from scipy.signal import cheby2
from .filterbox import FilterBox


class Cheby2Box(FilterBox):

    _filter_properties = ["manual", "broadband", "self_attenuatable"]

    def _get_filter_name(self):
        return "cheby2"

    def get_coefficients(self):
        N = self.get_order()
        Wn = self._get_std_passband()
        attenuation = self.get_min_attenuation()
        btype = self._get_btype()
        b, a = cheby2(N, attenuation, Wn, btype)
        return b, a
