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
    __synchronization_signal = None
    __synchronization_psd = None
    __synchronization_peaks = None

    def _copy_data(self, data):
        times = data[0]
        signal = data[1]
        synchronization_signal = data[2]
        self.__do_fft(times, signal, synchronization_signal)

    def _load_imaging_data_from_plotter(self, reader):
        times = reader.times
        signal = reader.averaged_signal
        frames = np.arange(0, times.size)
        regress_result = linregress(frames, times)
        dt = regress_result.slope
        times = frames * dt
        synchronization_signal = reader.synchronization_signal
        self.__do_fft(times, signal, synchronization_signal)

    def __do_fft(self, times, signal, synchronization_signal):
        spectrum = np.abs(fft(signal))
        F = 1000.0 / np.diff(times).mean()
        frequencies = np.arange(0, spectrum.size) * F / spectrum.size
        idx = frequencies < 0.5 * F
        frequencies = frequencies[idx]
        spectrum = spectrum[idx]

        synchronization_spectrum = np.abs(fft(synchronization_signal))[idx]
        i0 = synchronization_spectrum.argmax()
        F0 = frequencies[i0]
        self.__synchronization_peaks = np.arange(1, 11) * F0

        self.__times = times
        self.__values = signal
        self.__frequencies = frequencies
        self.__spectrum = spectrum
        self.__synchronization_signal = synchronization_signal
        self.__synchronization_psd = synchronization_spectrum

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

    def get_time(self):
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

    def get_synchronization_signal(self):
        """
        Returns the temporal dependency of the reference cosine
        which in term reflects how stimulus changes during the time
        """
        return self.__synchronization_signal

    def get_synchronization_psd(self):
        """
        Returns spectrum of the reference signal. Such spectrum reflects at which frequency the signal oscillates
        """
        return self.__synchronization_psd

    def get_synchronization_peaks(self):
        """
        Returns frequencies at which stimulus oscillations may contribute to the trace spectrum, i.e.:
        F0, 2F0, ..., 10F0,
        where F0 is a frequency at which the stimulus oscillates
        """
        return self.__synchronization_peaks

    def _save_data(self, npz_filename):
        if self.__times is not None:
            np.savez(npz_filename,
                     time=self.get_times(),
                     average_signal=self.get_values(),
                     frequency=self.get_frequencies(),
                     average_psd=self.get_spectrum(),
                     synchronization_signal=self.get_synchronization_signal(),
                     synchronization_peaks=self.get_synchronization_peaks(),
                     synchronization_psd=self.get_synchronization_psd())

    def _get_data_to_save(self):
        return {
            "TIME": self.get_times(),
            "AVERAGE_SIGNAL": self.get_values(),
            "FREQUENCY": self.get_frequencies(),
            "AVERAGE_PSD": self.get_spectrum(),
            "SYNCHRONIZATION_SIGNAL": self.get_synchronization_signal(),
            "SYNCHRONIZATION_PEAKS": self.get_synchronization_peaks(),
            "SYNCHRONIZATION_PSD": self.get_synchronization_psd()
        }

    def _load_data(self, npz_filename):
        file_data = np.load(npz_filename)
        self.__times = file_data['time']
        self.__values = file_data['average_signal']
        self.__frequencies = file_data["frequency"]
        self.__spectrum = file_data["average_psd"]
        self.__synchronization_signal = file_data["synchronization_signal"]
        self.__synchronization_psd = file_data["synchronization_psd"]
        self.__synchronization_peaks = file_data["synchronization_peaks"]

    def get_sample_rate(self):
        """
        Returns sample rate of the imaging signal in Hz
        """
        time_diffs = np.diff(self.get_time()).mean()
        return 1000/time_diffs
