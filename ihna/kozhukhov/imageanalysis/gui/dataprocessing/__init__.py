# -*- coding: utf-8

from .spatialfilterdlg import SpatialFilterDlg
from .MapAverageDlg import MapAverageDlg
from .mapfillerdlg import MapFillerDlg
from .tracefilter import TraceFilterDlg
from .tracepower import TracePower
from .setmainprocessor import SetMainProcessor
from .amplitudemap import AmplitudeMap
from .phasemap import PhaseMap
from .complexmap import ComplexMap
from .cutmap import CutMap
from .mapcorrelationprocessor import MapCorrelationProcessor
from .pinwheelselector import PinwheelSelector


def get_data_processors(parent):
    from ihna.kozhukhov.imageanalysis.gui.mapresultlistdlg import MapResultListDlg
    from ihna.kozhukhov.imageanalysis.gui.tracesresultlistdlg import TraceResultListDlg

    processors = {}
    if isinstance(parent, MapResultListDlg):
        processors.update({
            "Map filter": SpatialFilterDlg,
            "Compute average value": MapAverageDlg,
            "New map with predefined values": MapFillerDlg,
            "Set ROI using this map": SetMainProcessor,
            "Get amplitude map": AmplitudeMap,
            "Get phase map": PhaseMap,
            "Get complex map": ComplexMap,
            "Cut map": CutMap,
            "Map correlation": MapCorrelationProcessor,
            "Pinwheel selector": PinwheelSelector
        })

    if isinstance(parent, TraceResultListDlg):
        processors.update({
            "Trace filter": TraceFilterDlg,
            "Trace power": TracePower
        })

    return processors
