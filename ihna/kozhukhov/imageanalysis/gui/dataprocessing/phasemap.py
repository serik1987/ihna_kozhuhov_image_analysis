# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.gui.phasemapviewer import PhaseMapViewer
from .datatodataprocessor import DataToDataProcessor


class PhaseMap(DataToDataProcessor):

    def _get_default_minor_name(self):
        return "phase"

    def _place_additional_options(self, parent):
        return None

    def _process(self):
        self._output_data = self._input_data.phase_map()

    def _get_result_viewer(self):
        return PhaseMapViewer

    def _check_input_data(self):
        if not self._input_data.is_complex():
            raise ValueError("Please, select any complex map")
