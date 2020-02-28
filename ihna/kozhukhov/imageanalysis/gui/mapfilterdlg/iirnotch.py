# -*- coding: utf-8

import wx
from .filterbox import FilterBox


class IirNotchBox(FilterBox):

    def _get_filter_name(self):
        return "iirnotch"
