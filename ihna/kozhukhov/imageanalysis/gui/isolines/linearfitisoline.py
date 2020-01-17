# -*- coding: utf-8

from ihna.kozhukhov.imageanalysis.sourcefiles import StreamFileTrain
from ihna.kozhukhov.imageanalysis.isolines import LinearFitIsoline
from .isoline import IsolineEditor


class LinearFitIsolineEditor(IsolineEditor):
    """
    Represents widgets for creating and setting properties of the LinearFitIsoline
    """

    def get_name(self) -> str:
        return "Linear fit"

    def __init__(self, parent, train: StreamFileTrain, selector):
        super().__init__(parent, train, selector)

    def create_isoline(self, train, sync):
        isoline = LinearFitIsoline(train, sync)
        return isoline
