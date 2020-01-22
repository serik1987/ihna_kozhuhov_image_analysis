# -*- coding: utf-8

import numpy as np
from scipy import diff
from scipy.fftpack import fft


class TraceProcessor:
    """
    Provides an interface for post-processing of traces after they have been read
    The trace postprocessing is based on SCIPY
    """

    __init_frame = None
    __final_frame = None
    __time_arrivals = None
    __synch_channels = None
    __data_not_removed = None
    __isolines = None
    __data = None
    __reference_signal = None
    __accepted_points = 96

    def __init__(self, reader, isoline, sync):
        """
        Arguments:
             reader - instance of the TraceReaderAndCleaner. Note, that at least 'TIME' channel shall be read
             isoline - instance of the Isoline that you used for the isoline remove during the trace read
             sync - instance of Synchronization that you used for trace reading and cleaning
        """
        self.__init_frame = reader.initial_frame
        self.__final_frame = reader.final_frame
        self.__synch_channels = []
        self.__data = []
        self.__data_not_removed = []
        self.__isolines = []
        data_raw = reader.traces_before_remove
        isolines = reader.isolines
        idx = 0
        for pixel in reader.pixels:
            if pixel[0] == 'TIME':
                self.__time_arrivals = reader.get_trace(idx)
            elif pixel[0] == 'SYNC':
                self.__synch_channels.append(reader.get_trace(idx))
            else:
                self.__data_not_removed.append(data_raw[:, idx].reshape((reader.frame_number, 1)))
                self.__isolines.append(isolines[:, idx].reshape((reader.frame_number, 1)))
                self.__data.append(reader.get_trace(idx).reshape((reader.frame_number, 1)))
            idx += 1
        self.__data_not_removed = np.hstack(self.__data_not_removed)
        self.__isolines = np.hstack(self.__isolines)
        self.__data = np.hstack(self.__data)
        self.__reference_signal = sync.reference_sin
        cycles = isoline.analysis_final_cycle - isoline.analysis_initial_cycle + 1
        self.__accepted_points = cycles * 10

    def get_frame_lim(self):
        """
        Returns a tuple containing an initial frame and a final frame
        """
        return self.__init_frame, self.__final_frame

    def get_frame_vector(self):
        return np.arange(self.__init_frame, self.__final_frame+1)

    def get_time_arrivals(self):
        """
        Returns a vector containing time arrivals or None if the time vector is not included into the analysis
        """
        return self.__time_arrivals

    def get_synch_channel_number(self):
        """
        Returns total number of synchronization channels written
        """
        return len(self.__synch_channels)

    def get_synch_channel(self, chan):
        """
        Returns the data from a certain synchronization channel

        Arguments:
            chan - number of this synchronization channel
        """
        return self.__synch_channels[chan]

    def get_data_not_removed(self):
        return self.__data_not_removed

    def get_isolines(self):
        return self.__isolines

    def get_data(self):
        return self.__data

    def get_psd_not_removed(self):
        data = self.get_data_not_removed() - self.get_data_not_removed().mean()
        spectrum = fft(data, axis=0)[:self.__accepted_points]
        return np.abs(spectrum)

    def get_isoline_psd(self):
        data = self.get_isolines() - self.get_isolines().mean()
        spectrum = fft(data, axis=0)[:self.__accepted_points]
        return np.abs(spectrum)

    def get_psd(self):
        data = self.get_data() - self.get_data().mean()
        spectrum = fft(data, axis=0)[:self.__accepted_points]
        return np.abs(spectrum)

    def get_reference_signal(self):
        return self.__reference_signal

    def get_average_signal(self):
        return self.get_data().mean(axis=1)

    def get_median_signal(self):
        return np.median(self.get_data(), axis=1)

    def get_average_signal_spectrum(self):
        """
        First, averages the signal.
        Next, plots the spectrum of the averaged signal
        """
        data = self.get_average_signal()
        data -= data.mean()
        spectrum = fft(data, axis=0)[:self.__accepted_points]
        return np.abs(spectrum)

    def get_median_signal_spectrum(self):
        """
        First, computes the median of the signal
        Next, plots the spectrum of the result
        """
        data = self.get_median_signal()
        data -= data.mean()
        spectrum = fft(data, axis=0)[:self.__accepted_points]
        return np.abs(spectrum)

    def get_average_spectrum(self):
        """
        First, plots the spectrum of the signal
        Next, calculates its mean
        """
        return self.get_psd().mean(axis=1)

    def get_median_spectrum(self):
        """
        First, plots the spectrum of the signal
        Next, calculates its median
        """
        return np.median(self.get_psd(), axis=1)

    def get_reference_spectrum(self):
        """
        Plots the spectrum of the reference signal
        """
        data = self.get_reference_signal()
        data -= data.mean()
        spectrum = fft(data)[:self.__accepted_points]
        return np.abs(spectrum)
