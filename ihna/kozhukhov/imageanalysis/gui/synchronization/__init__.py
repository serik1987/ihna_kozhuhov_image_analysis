# -*- coding: utf-8

from .synchronization import SynchronizationEditor
from .externalsynchronization import ExternalSynchronizationEditor
from .nosynchronization import NoSynchronizationEditor
from .quasistimulussynchronization import QuasiStimulusSynchronizationEditor
from .quasitimesynchronization import QuasiTimeSynchronizationEditor


synchronization_editors = [NoSynchronizationEditor, QuasiStimulusSynchronizationEditor,
                           QuasiTimeSynchronizationEditor, ExternalSynchronizationEditor]
