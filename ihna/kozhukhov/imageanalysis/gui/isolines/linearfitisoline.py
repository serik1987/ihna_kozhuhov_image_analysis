# -*- coding: utf-8


from .isoline import IsolineEditor
from ihna.kozhukhov.imageanalysis.sourcefiles import StreamFileTrain


class LinearFitIsolineEditor(IsolineEditor):
    """
    Represents widgets for creating and setting properties of the LinearFitIsoline
    """

    def get_name(self) -> str:
        return "Linear fit"

    def __init__(self, parent, train: StreamFileTrain):
        super().__init__(parent, train)
