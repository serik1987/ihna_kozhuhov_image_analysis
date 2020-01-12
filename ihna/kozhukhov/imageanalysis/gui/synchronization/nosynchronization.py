# -*- coding: utf-8

from .synchronization import SynchronizationEditor


class NoSynchronizationEditor(SynchronizationEditor):
    """
    Represents the set of widgets that allows to create an instance of NoSynchronization class
    """

    def get_name(self):
        return "No synchronization"

    def __init__(self, parent, train):
        super().__init__(parent, train)
