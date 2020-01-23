# -*- coding: utf-8

import numpy as np
from scipy import diff
from scipy.fftpack import fft
from ihna.kozhukhov.imageanalysis.synchronization import NoSynchronization, QuasiStimulusSynchronization, \
    QuasiTimeSynchronization, ExternalSynchronization
from ihna.kozhukhov.imageanalysis.isolines import NoIsoline, LinearFitIsoline, TimeAverageIsoline


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

    __train_properties = None
    __synchronization_properties = None
    __isoline_properties = None
    __trace_reading_properties = None

    __average_strategy = "psd_than_average"
    __average_method = "mean"

    def __init__(self, reader, isoline, sync, train):
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

        self.__set_train_properties(train)
        self.__set_sync_properties(sync)
        self.__set_isoline_properties(isoline)

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
        """
        Returns the individual traces before the isoline removal
        """
        return self.__data_not_removed

    def get_isolines(self):
        """
        Returns the isolines
        """
        return self.__isolines

    def get_data(self):
        """
        Returns the individual traces
        """
        return self.__data

    def get_psd_not_removed(self):
        """
        Returns Power Spectral Densities of individual signals before the isoline remove
        """
        data = self.get_data_not_removed() - self.get_data_not_removed().mean()
        spectrum = fft(data, axis=0)[:self.__accepted_points]
        return np.abs(spectrum)

    def get_isoline_psd(self):
        """
        Returns the Power Spectral Densities of individual isolines
        """
        data = self.get_isolines() - self.get_isolines().mean()
        spectrum = fft(data, axis=0)[:self.__accepted_points]
        return np.abs(spectrum)

    def get_psd(self):
        """
        Returns the Power Spectral Density of the signals from individual pixels
        """
        data = self.get_data() - self.get_data().mean()
        spectrum = fft(data, axis=0)[:self.__accepted_points]
        return np.abs(spectrum)

    def get_reference_signal(self):
        """
        Returns the reference signal
        """
        return self.__reference_signal

    def get_average_signal(self):
        """
        Returns the average of the signal across all pixels within the ROI
        """
        return self.get_data().mean(axis=1)

    def get_median_signal(self):
        """
        Returns the median of the source signal across all pixels within the ROI
        """
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

    def __set_train_properties(self, train):
        self.__train_properties = {
            "filename": train.file_path + train.filename,
            "file_number": train.file_number,
            "frame_number": train.total_frames,
            "frame_dimensions": np.array((train.frame_shape[1], train.frame_shape[0])),
            "pixel_number": train.frame_size,
            "frame_header_size": train.frame_header_size,
            "frame_body_size": train.frame_image_size,
            "frame_size": train.frame_size,
            "file_header_size": train.file_header_size,
            "experiment_mode": train.experiment_mode,
            "synch_channels": train.synchronization_channel_number
        }

    def __set_sync_properties(self, sync):
        self.__synchronization_properties = {
            "initial_frame": sync.initial_frame,
            "final_frame": sync.final_frame,
            "frame_number": sync.frame_number,
            "do_precise": sync.do_precise,
            "harmonic": sync.harmonic,
            "phase_increment": sync.phase_increment,
            "initial_phase": sync.initial_phase
        }
        if isinstance(sync, NoSynchronization):
            self.__synchronization_properties.update({
                "type": "no"
            })
        if isinstance(sync, ExternalSynchronization):
            self.__synchronization_properties.update({
                "type": "external",
                "synch_channel": sync.channel_number,
                "initial_cycle": sync.initial_cycle,
                "final_cycle": sync.final_cycle,
                "cycle_number": sync.final_cycle - sync.initial_cycle + 1
            })
        if isinstance(sync, QuasiStimulusSynchronization):
            self.__synchronization_properties.update({
                "type": "quasi-stimulus",
                "stimulus_period": sync.stimulus_period,
                "initial_cycle": sync.initial_cycle,
                "final_cycle": sync.final_cycle
            })
        if isinstance(sync, QuasiTimeSynchronization):
            self.__synchronization_properties.update({
                "type": "quasi-time",
                "stimulus_period": sync.stimulus_period,
                "initial_cycle": sync.initial_cycle,
                "final_cycle": sync.final_cycle
            })

    def __set_isoline_properties(self, isoline):
        self.__isoline_properties = {
            "analysis_initial_cycle": isoline.analysis_initial_cycle,
            "analysis_final_cycle": isoline.analysis_final_cycle,
            "analysis_cycle_number": isoline.analysis_final_cycle - isoline.analysis_initial_cycle + 1,
            "analysis_initial_frame": isoline.analysis_initial_frame,
            "analysis_final_frame": isoline.analysis_final_frame,
            "analysis_frame_number": isoline.analysis_final_frame - isoline.analysis_initial_frame + 1,
            "isoline_initial_cycle": isoline.isoline_initial_cycle,
            "isoline_final_cycle": isoline.isoline_final_cycle,
            "isoline_cycle_number": isoline.isoline_final_cycle - isoline.isoline_initial_cycle + 1,
            "isoline_initial_frame": isoline.isoline_initial_frame,
            "isoline_final_frame": isoline.isoline_final_frame,
            "isoline_frame_number": isoline.isoline_final_frame - isoline.isoline_initial_frame + 1
        }
        if isinstance(isoline, NoIsoline):
            self.__isoline_properties.update({
                "type": "no isoline"
            })
        if isinstance(isoline, LinearFitIsoline):
            self.__isoline_properties.update({
                "type": "linear fit"
            })
        if isinstance(isoline, TimeAverageIsoline):
            self.__isoline_properties.update({
                "type": "time average",
                "radius": isoline.average_cycles
            })

    def get_annotation_text(self):
        return """Analysis epoch: {0} - {1} cycles ({2} - {3} frames)
Isoline plotting epoch: {4} - {5} cycles ({6} - {7} frames)""".format(
            self.__isoline_properties['analysis_initial_cycle'],
            self.__isoline_properties['analysis_final_cycle'],
            self.__isoline_properties['analysis_initial_frame'],
            self.__isoline_properties['analysis_final_frame'],
            self.__isoline_properties['isoline_initial_cycle'],
            self.__isoline_properties['isoline_final_cycle'],
            self.__isoline_properties['isoline_initial_frame'],
            self.__isoline_properties['isoline_final_frame']
        )

    def set_average_strategy(self, value):
        """
        Sets the average strategy

        Arguments:
             value - one of the following possible strings:
                'average_that_psd' - first, average the signal, next, calculate the Power Spectrum Density
                'psd_than_average' - first, calculate the Power Spectrum Density, next, average the signal
        """
        if value in ['average_that_psd', 'psd_than_average']:
            self.__average_strategy = value
        else:
            raise ValueError("Unknown or unsupported average strategy")
