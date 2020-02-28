# -*- coding: utf-8

import wx
from .filterbox import FilterBox


class Cheby1Box(FilterBox):

    def _get_filter_name(self):
        return "cheby1"
