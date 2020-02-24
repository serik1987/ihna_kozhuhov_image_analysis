# -*- coding: utf-8

import wx
from .filterdlg import FilterDlg


class AnimalFilterDlg(FilterDlg):

    def __init__(self, parent, animal_filter):
        super().__init__(parent, animal_filter, "Animal filter")

    def get_property_captions(self):
        return {
            "specimen": "Specimen",
            "conditions": "Conditions",
            "recording_site": "Recording site"
        }
