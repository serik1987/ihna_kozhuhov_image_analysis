# -*- coding: utf-8

from .synchronization import SynchronizationEditor


class QuasiStimulusSynchronizationEditor(SynchronizationEditor):
    """
    Provides a GUI for creating QuasiStimulusSynchronization and setting its properties
    """

    def get_name(self):
        return "Quasi-stimulus synchronization"

    def __init__(self, parent, train, selector):
        super().__init__(parent, train, selector)
