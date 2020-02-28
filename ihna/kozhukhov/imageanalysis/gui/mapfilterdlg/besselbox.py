# -*- coding: utf-8

import wx
from .filterbox import FilterBox


class BesselBox(FilterBox):

    def _get_filter_name(self):
        return "bessel"
