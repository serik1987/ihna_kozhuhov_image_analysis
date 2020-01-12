# -*- coding: utf-8

from .synchronization import SynchronizationEditor
from .externalsynchronization import ExternalSynchronizationEditor
from .nosynchronization import NoSynchronizationEditor

synchronization_editors = [ExternalSynchronizationEditor]
