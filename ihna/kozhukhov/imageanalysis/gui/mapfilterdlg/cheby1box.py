# -*- coding: utf-8

import wx
from scipy.signal import cheby1
from .filterbox import FilterBox


class Cheby1Box(FilterBox):

    _filter_properties = ["broadband", "manual", "rippable"]

    def _get_filter_name(self):
        return "cheby1"

    def get_coefficients(self):
        N = self.get_order()
        Wn = self._get_std_passband()
        ripples = self.get_ripples()
        btype = self._get_btype()
        b, a = cheby1(N, ripples, Wn, btype)
        return b, a
