# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.gui.amplitudemapviewerdlg import AmplitudeMapViewerDlg
from .datatodataprocessor import DataToDataProcessor


class AmplitudeMap(DataToDataProcessor):

    def _get_default_minor_name(self):
        return "amplitude"

    def _place_additional_options(self, parent):
        return None

    def _process(self):
        self._output_data = self._input_data.amplitude_map()

    def _get_result_viewer(self):
        return AmplitudeMapViewerDlg

    def _check_input_data(self):
        if not self._input_data.is_complex_map():
            raise ValueError("Please, select any complex map")
