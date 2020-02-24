# -*- coding: utf-8

import wx
from .filterdlg import FilterDlg


class CaseFilterDlg(FilterDlg):

    def __init__(self, parent, case_filter):
        super().__init__(parent, case_filter, "Case Filter")

    def get_property_captions(self):
        return {
            "short_name": "Short name",
            "long_name": "Long name",
            "stimulation": "Stimulation",
            "additional_stimulation": "Additional stimulation",
            "special_conditions": "Special conditions",
            "additional_information": "Additional information"
        }
