# -*- coding: utf-8

import wx
from .filterbox import FilterBox


class ButterBox(FilterBox):

    def _get_filter_name(self):
        return "butter"
