# -*- coding: utf-8

from .synchronization import SynchronizationEditor


class ExternalSynchronizationEditor(SynchronizationEditor):
    """
    The editor provides a GUI for creating the ExternalSynchronization object and set all its necessary properties
    """

    def get_name(self):
        return "Synchronization by the external signal"

    def __super__(self, parent, train, selector):
        super().__init__(parent, train, selector)
