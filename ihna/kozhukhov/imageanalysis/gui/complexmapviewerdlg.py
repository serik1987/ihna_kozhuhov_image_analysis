# -*- coding: utf-8

import numpy as np
from .mapviewerdlg import MapViewerDlg


class ComplexMapViewerDlg(MapViewerDlg):

    def _plot_graphs(self, figure, data):
        amplitude_axes = figure.add_subplot(121)
        amplitude_map = amplitude_axes.imshow(np.abs(data.get_data()), cmap="gray")
        figure.colorbar(amplitude_map, ax=amplitude_axes)
        amplitude_axes.set_title("Amplitude map")
        phase_axes = figure.add_subplot(122)
        H = data.get_harmonic()
        phase_data = np.angle(data.get_data()) / H
        phase_data[phase_data < 0] += 2 * np.pi / H
        phase_data = 180 * phase_data / np.pi
        phase_max = 360 / H
        phase_map = phase_axes.imshow(phase_data, cmap="hsv", vmin=0, vmax=phase_max)
        phase_axes.set_title("Phase map")
        figure.colorbar(phase_map, ax=phase_axes)
