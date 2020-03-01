# -*- coding: utf-8

from scipy.signal import ellip
from .filterbox import FilterBox


class EllipBox(FilterBox):

    _filter_properties = ["broadband", "manual", "rippable", "self_attenuatable"]

    def _get_filter_name(self):
        return "ellip"

    def get_coefficients(self):
        N = self.get_order()
        Wn = self._get_std_passband()
        rp = self.get_ripples()
        rs = self.get_min_attenuation()
        btype = self._get_btype()
        b, a = ellip(N, rs, rp, Wn, btype)
        return b, a
