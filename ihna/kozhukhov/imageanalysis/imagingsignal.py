# -*- coding: utf-8

import numpy as np
from scipy.stats import linregress
from scipy.fftpack import fft
from .imagingdata import ImagingData


class ImagingSignal(ImagingData):

    __times = None
    __values = None
    __frequencies = None
    __spectrum = None

    def _load_imaging_data_from_plotter(self, reader):
        times = reader.times
        signal = reader.averaged_signal
        frames = np.arange(0, times.size)
        regress_result = linregress(frames, times)
        dt = regress_result.slope
        times = frames * dt
        spectrum = np.abs(fft(signal))
        F = 1000.0 / dt
        frequencies = np.arange(0, spectrum.size) * F / spectrum.size
        idx = frequencies < 0.5 * F
        frequencies = frequencies[idx]
        spectrum = spectrum[idx]

        self.__times = times
        self.__values = signal
        self.__frequencies = frequencies
        self.__spectrum = spectrum

    def get_data(self):
        """
        Returns tuple containing all times and all values assigned to the graph
        """
        return self.__times, self.__values

    def get_times(self):
        """
        Returns vector containing times (in ms)
        """
        return self.__times

    def get_values(self):
        """
        Returns vector containing values from the imaging signal
        """
        return self.__values

    def get_frequencies(self):
        """
        Returns vector containing frequencies which spectrum values were calculated during the FFT (in Hz)
        """
        return self.__frequencies

    def get_spectrum(self):
        """
        Returns the power spectrum values
        """
        return self.__spectrum
