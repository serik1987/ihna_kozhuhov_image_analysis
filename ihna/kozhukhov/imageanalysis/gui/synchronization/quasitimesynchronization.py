# -*- coding: utf-8

from .synchronization import SynchronizationEditor


class QuasiTimeSynchronizationEditor(SynchronizationEditor):
    """
    Represents an editor that allows to create QuasiTimeSynchronization and set all its parameters
    """

    def get_name(self):
        return "Quasi-time synchronization"

    def __init__(self, parent, train, selector):
        super().__init__(parent, train, selector)
