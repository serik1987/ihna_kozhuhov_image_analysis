# -*- coding: utf-8

import numpy as np
import wx
from .mapviewerdlg import MapViewerDlg


class PhaseMapViewer(MapViewerDlg):

    def _plot_graphs(self, figure, data):
        d = data.get_data() * 180 / np.pi
        h = data.get_harmonic()
        d[d < 0] += 360 / h
        ax = figure.add_subplot(111)
        img = ax.imshow(d, cmap='hsv', vmin=0, vmax=360/h)
        figure.colorbar(img, ax=ax)
