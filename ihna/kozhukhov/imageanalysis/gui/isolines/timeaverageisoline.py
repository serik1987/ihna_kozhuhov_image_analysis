# -*- coding: utf-8

from .isoline import IsolineEditor
from ihna.kozhukhov.imageanalysis.sourcefiles import StreamFileTrain


class TimeAverageIsolineEditor(IsolineEditor):
    """
    Represents widgets to select the TimeAverageIsoline and set all its properties
    """

    def get_name(self) -> str:
        return "Time average"

    def __init__(self, parent, train: StreamFileTrain):
        super().__init__(parent, train)
