# -*- coding: utf-8

from scipy.signal import filtfilt
from ihna.kozhukhov.imageanalysis import ImagingSignal


def filter_trace(input_trace, b, a):
    time_vector = input_trace.get_times()
    input_values = input_trace.get_values()
    output_values = filtfilt(b, a, input_values)
    output_data = time_vector, output_values, input_trace.get_synchronization_signal()
    output_trace = ImagingSignal(input_trace.get_features(), output_data)
    output_trace.get_features().update({
        "minor_name": "tracefilt",
        "original_trace": input_trace.get_full_name()
    })

    return output_trace
