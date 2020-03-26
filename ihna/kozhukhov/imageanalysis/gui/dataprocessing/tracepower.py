# -*- coding: utf-8

import numpy as np
from ihna.kozhukhov.imageanalysis import ImagingSignal
from .datatonumberprocessor import DataToNumberProcessor


class TracePower(DataToNumberProcessor):

    def _get_processor_title(self):
        return "Trace power"

    def _check_input_data(self):
        super()._check_input_data()
        if not isinstance(self._input_data, ImagingSignal):
            raise ValueError("This processor can process traces only")

    def _get_default_minor_name(self):
        return "power"

    def _place_additional_options(self, parent):
        pass

    def _process(self):
        data = self._input_data
        signal = data.get_values()
        signal_power = np.sqrt((signal**2).sum())
        self._output_data = signal_power

    def _print_output_value(self):
        return "Signal power, %%: %1.2f" % self._output_data
