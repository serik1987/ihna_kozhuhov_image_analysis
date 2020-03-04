# -*- coding: utf-8

import numpy as np
from .mapviewerdlg import MapViewerDlg


class AmplitudeMapViewerDlg(MapViewerDlg):

    def _plot_graphs(self, figure, data):
        ax = figure.add_subplot(111)
        imaging_data = ax.imshow(data.get_data(), cmap="gray")
        figure.colorbar(imaging_data, ax=ax)
