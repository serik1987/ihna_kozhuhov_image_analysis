# -*- coding: utf-8

from ihna.kozhukhov.imageanalysis.sourcefiles import StreamFileTrain
from ihna.kozhukhov.imageanalysis.isolines import NoIsoline
from .isoline import IsolineEditor


class NoIsolineEditor(IsolineEditor):
    """
    Manages widgets that shall be used to select no isolines
    """

    def get_name(self):
        return "Don't remove isoline"

    def __init__(self, parent, train: StreamFileTrain, selector):
        super().__init__(parent, train, selector)

    def is_first(self):
        return True

    def create_isoline(self, train, sync):
        isoline = NoIsoline(train, sync)
        return isoline
