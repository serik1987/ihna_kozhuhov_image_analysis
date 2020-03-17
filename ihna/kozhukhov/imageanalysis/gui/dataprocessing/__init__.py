# -*- coding: utf-8

from .spatialfilterdlg import SpatialFilterDlg
from .MapAverageDlg import MapAverageDlg
from .mapfillterdlg import MapFillterDlg


def get_data_processors(parent):
    from ihna.kozhukhov.imageanalysis.gui.mapresultlistdlg import MapResultListDlg

    processors = {}
    if isinstance(parent, MapResultListDlg):
        processors.update({
            "Map filter": SpatialFilterDlg,
            "Compute average value": MapAverageDlg,
            "New map with predefined valued": MapFillterDlg
        })

    return processors
