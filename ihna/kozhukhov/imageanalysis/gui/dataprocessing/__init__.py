# -*- coding: utf-8

from .spatialfilterdlg import SpatialFilterDlg
from .MapAverageDlg import MapAverageDlg
from .mapfillerdlg import MapFillerDlg
from .tracefilter import TraceFilterDlg


def get_data_processors(parent):
    from ihna.kozhukhov.imageanalysis.gui.mapresultlistdlg import MapResultListDlg
    from ihna.kozhukhov.imageanalysis.gui.tracesresultlistdlg import TraceResultListDlg

    processors = {}
    if isinstance(parent, MapResultListDlg):
        processors.update({
            "Map filter": SpatialFilterDlg,
            "Compute average value": MapAverageDlg,
            "New map with predefined valued": MapFillerDlg
        })

    if isinstance(parent, TraceResultListDlg):
        processors.update({
            "Trace filter": TraceFilterDlg
        })

    return processors
