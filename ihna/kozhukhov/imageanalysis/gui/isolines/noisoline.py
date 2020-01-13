# -*- coding: utf-8

from .isoline import IsolineEditor


class NoIsolineEditor(IsolineEditor):
    """
    Manages widgets that shall be used to select no isolines
    """

    def get_name(self):
        return "Don't remove isoline"

    def __init__(self, parent, train):
        super().__init__(parent, train)
