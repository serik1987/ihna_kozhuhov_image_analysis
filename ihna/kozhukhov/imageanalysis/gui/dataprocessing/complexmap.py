# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis import ImagingMap
from ihna.kozhukhov.imageanalysis.gui.complexmapviewerdlg import ComplexMapViewerDlg
from .twodatatodataprocessor import TwoDataToDataProcessor


class ComplexMap(TwoDataToDataProcessor):

    def _get_default_minor_name(self):
        return "complex"

    def _place_additional_options(self, parent):
        return None

    def _check_two_maps(self):
        if self._input_data.is_amplitude_map() and self._second_map.is_phase_map():
            return
        raise ValueError("The first map must be amplitude while the second map must be phase")

    def _process(self):
        self._output_data = ImagingMap.complex_map(self._input_data, self._second_map)

    def _get_result_viewer(self):
        return ComplexMapViewerDlg
