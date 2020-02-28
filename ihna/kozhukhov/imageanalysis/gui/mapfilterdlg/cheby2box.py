# -*- coding: utf-8

import wx
from .filterbox import FilterBox


class Cheby2Box(FilterBox):

    def _get_filter_name(self):
        return "cheby2"
