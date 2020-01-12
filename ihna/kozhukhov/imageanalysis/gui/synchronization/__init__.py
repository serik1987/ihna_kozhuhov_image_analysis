# -*- coding: utf-8

from .synchronization import SynchronizationEditor
from .externalsynchronization import ExternalSynchronizationEditor
from .nosynchronization import NoSynchronizationEditor
from .quasistimulussynchronization import QuasiStimulusSynchronizationEditor

synchronization_editors = [ExternalSynchronizationEditor, NoSynchronizationEditor, QuasiStimulusSynchronizationEditor]
