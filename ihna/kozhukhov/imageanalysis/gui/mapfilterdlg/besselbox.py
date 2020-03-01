# -*- coding: utf-8

import wx
from scipy.signal import bessel
from .filterbox import FilterBox


class BesselBox(FilterBox):

    _filter_properties = ["broadband", "manual"]

    def _get_filter_name(self):
        return "bessel"

    def get_coefficients(self):
        N = self.get_order()
        Wn = self._get_std_passband()
        btype = self._get_btype()
        b, a = bessel(N, Wn, btype)
        return b, a
