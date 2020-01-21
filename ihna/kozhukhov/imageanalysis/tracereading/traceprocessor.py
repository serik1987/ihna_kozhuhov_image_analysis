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
    __data = None
    __reference_signal = None

    def __init__(self, reader, sync):
        """
        Arguments:
             reader - instance of the TraceReaderAndCleaner. Note, that at least 'TIME' channel shall be read
             sync - instance of Synchronization that you used for trace reading and cleaning
        """
        self.__init_frame = reader.initial_frame
        self.__final_frame = reader.final_frame
        self.__synch_channels = []
        self.__data = []
        idx = 0
        for pixel in reader.pixels:
            if pixel[0] == 'TIME':
                self.__time_arrivals = reader.get_trace(idx)
            elif pixel[0] == 'SYNC':
                self.__synch_channels.append(reader.get_trace(idx))
            else:
                self.__data.append(reader.get_trace(idx).reshape((reader.frame_number, 1)))
            idx += 1
        self.__data = np.hstack(self.__data)
        self.__reference_signal = sync.reference_sin

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

    def get_data(self):
        return self.__data

    def get_psd(self):
        data = self.get_data()
        spectrum = fft(data, n=256, axis=0)
        return np.abs(spectrum)

    def get_reference_signal(self):
        return self.__reference_signal
