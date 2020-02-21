# -*- coding: utf-8

import wx
from .frameaccumulatordlg import FrameAccumulatorDlg
from ihna.kozhukhov.imageanalysis.accumulators import MapPlotter


class MapPlotterDlg(FrameAccumulatorDlg):

    def __init__(self, parent, train):
        super().__init__(parent, train, "Map Plotter")

    def _get_accumulator_class(self):
        return MapPlotter
