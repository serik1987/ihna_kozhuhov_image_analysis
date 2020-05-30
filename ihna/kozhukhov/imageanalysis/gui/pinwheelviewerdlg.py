# -*- coding: utf-8

from .mapviewerdlg import MapViewerDlg


class PinwheelViewerDlg(MapViewerDlg):

    def _plot_graphs(self, figure, data):
        pwc_list = data.getCenterList()
        width = data.getMapWidth()
        height = data.getMapHeight()
        ax = figure.add_subplot(111)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        [ax.plot(x, y, 'r*') for x, y in pwc_list]
