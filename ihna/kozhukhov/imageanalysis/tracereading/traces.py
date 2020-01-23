# -*- coding: utf-8

import os
import numpy as np
import scipy.io


class Traces:
    """
    Copies traces from the Trace Reader and provides their processing. The resultant signal contains
    only temporary-dependent data from the synchronization channel and from the pixels given at a frame

    Construction:
    Traces(case) - load the traces from the hard disk after they have been processed
    Traces(case, processor) - get the traces from the processor
    """

    __animal_name = None
    __case_name = None
    __prefix_name = None
    __postfix_name = None
    __roi_name = None
    __train_properties = None
    __synchronization_properties = None
    __isoline_properties = None
    __times = None
    __traces = None
    __frequencies = None
    __spectrums = None
    __avg_trace = None
    __avg_psd = None
    __reference = None
    __reference_psd = None

    def __init__(self):
        pass

    def get_animal_name(self):
        """
        Returns the cat name
        """
        if self.__animal_name is None:
            raise AttributeError("Please, set the cat's name by application of set_cat_name")
        else:
            return self.__animal_name

    def get_case_name(self):
        """
        Returns the case name
        """
        if self.__case_name is None:
            raise AttributeError("Please, set the case name")
        else:
            return self.__case_name

    def set_animal_name(self, value):
        """
        Sets the cat's name
        """
        self.__animal_name = value

    def set_case_name(self, value):
        """
        Sets the case name
        """
        self.__case_name = value

    def get_prefix_name(self):
        """
        Returns the prefix name or empty string if the prefix name is not defined
        """
        if self.__prefix_name is None:
            return ""
        else:
            return self.__prefix_name

    def get_postfix_name(self):
        """
        Returns the postfix name or empty string if the postfix name is not defined
        """
        if self.__postfix_name is None:
            return ""
        else:
            return self.__postfix_name

    def set_prefix_name(self, value):
        """
        Sets the prefix name
        """
        self.__prefix_name = value

    def set_postfix_name(self, value):
        """
        Sets the postfix name
        """
        self.__postfix_name = value

    def get_roi_name(self):
        """
        Returns the ROI name
        """
        if self.__roi_name is None:
            raise AttributeError("Please, specify the ROI name by application of set_roi_name()")
        else:
            return self.__roi_name

    def set_roi_name(self, value):
        """
        Sets the ROI name
        """
        self.__roi_name = value

    def get_fullname(self):
        """
        Returns the full name of the train
        """
        return "{0}_{1}{2}{3}_traces({4})".format(
            self.get_animal_name(),
            self.get_prefix_name(),
            self.get_case_name(),
            self.get_postfix_name(),
            self.get_roi_name()
        )

    def get_train_properties(self):
        """
        Returns the train properties
        """
        if self.__train_properties is None:
            raise AttributeError("Please, set the train properties by means of set_train_properties")
        else:
            return self.__train_properties

    def set_train_properties(self, value):
        """
        Sets the train properties
        """
        if isinstance(value, dict):
            self.__train_properties = value
        else:
            raise ValueError("Train properties shall be dictionary")

    def get_synchronization_properties(self):
        """
        Returns the synchronization properties
        """
        if self.__synchronization_properties is None:
            raise AttributeError("Please, apply set_synchronization_properties")
        else:
            return self.__synchronization_properties

    def set_synchronization_properties(self, value):
        """
        Sets the synchronization properties
        """
        if isinstance(value, dict):
            self.__synchronization_properties = value
        else:
            raise ValueError("Synchronization properties shall be dictionary")

    def get_isoline_properties(self):
        """
        Returns the isoline properties
        """
        if self.__isoline_properties is None:
            raise AttributeError("Please, set_isoline_properties()")
        else:
            return self.__isoline_properties

    def set_isoline_properties(self, value):
        """
        Sets the isoline properties
        """
        if isinstance(value, dict):
            self.__isoline_properties = value
        else:
            raise ValueError("Isoline properties shall be dictionary")

    def get_times(self):
        """
        Sets the times vector
        """
        if self.__times is None:
            raise AttributeError("Returns times")
        else:
            return self.__times

    def set_times(self, value):
        """
        Sets the times vector
        """
        self.__times = value

    def get_traces(self):
        """
        Returns the traces matrix
        (Columns correspond to appropriate pixels, rows correspond to timestamps,
        use get_times() to return time values that correspond to these timestamps)
        """
        if self.__traces is None:
            raise AttributeError("Please, set_traces()")
        else:
            return self.__traces

    def set_traces(self, value):
        """
        Returns a certain traces
        """
        self.__traces = value

    def get_frequencies(self):
        """
        Returns the frequencies vector
        """
        if self.__frequencies is None:
            raise AttributeError("Please, set_frequencies()")
        else:
            return self.__frequencies

    def set_frequencies(self, value):
        """
        Sets the frequencies vector. While spectrum values shall be plotted on axis Y,
        frequency values shall be plotted on axis X
        """
        self.__frequencies = value

    def get_spectrums(self):
        """
        Returns the spectrum of individual traces
        """
        if self.__spectrums is None:
            raise AttributeError("Please, set_spectrums()")
        else:
            return self.__spectrums

    def set_spectrums(self, value):
        """
        Sets the spectrum of individual traces
        """
        self.__spectrums = value

    def get_avg_trace(self):
        """
        Returns the average across all traces
        """
        if self.__avg_trace is None:
            raise AttributeError("Please, set_avg_trace()")
        else:
            return self.__avg_trace

    def set_avg_trace(self, value):
        """
        Sets the average across all traces
        """
        self.__avg_trace = value

    def get_avg_psd(self):
        """
        Returns the averaged spectrum
        """
        if self.__avg_psd is None:
            raise AttributeError("Please, set_avg_psd()")
        else:
            return self.__avg_psd

    def set_avg_psd(self, value):
        """
        Sets the averaged spectrum
        """
        self.__avg_psd = value

    def get_reference(self):
        """
        Returns the reference signal
        """
        if self.__reference is None:
            raise AttributeError("Please, set the reference signal")
        else:
            return self.__reference

    def get_reference_psd(self):
        """
        Returns PSD of the reference signal
        """
        if self.__reference_psd is None:
            raise AttributeError("Please, set the reference PSD")
        else:
            return self.__reference_psd

    def set_reference_signal(self, value):
        self.__reference = value

    def set_reference_psd(self, value):
        self.__reference_psd = value

    def __str__(self):
        try:
            s0 = """
Traces info
-------------------------------------------------------
Animal name: {0}
Case name: {1}
Prefix name: {2}
Postfix name: {3}
ROI name: {4}
Fullname: {5}
""".format(
                self.get_animal_name(),
                self.get_case_name(),
                self.get_prefix_name(),
                self.get_postfix_name(),
                self.get_roi_name(),
                self.get_fullname()
            )

            s1 = "Train properties:\n"
            for key, value in self.get_train_properties().items():
                s1 += "{0}: {1}\n".format(key, value)

            s2 = "Synchronization properties:\n"
            for key, value in self.get_synchronization_properties().items():
                s2 += "{0}: {1}\n".format(key, value)

            s3 = "Isoline properties:\n"
            for key, value in self.get_isoline_properties().items():
                s3 += "{0}: {1}\n".format(key, value)

            s4 = """Other values:
Times: {0} {8}
Traces: {1} {9}
Avg trace: {4} {10}
Frequencies: {2} {11}
Spectrums: {3} {12}
Avg psd: {5} {13}
Reference: {6} {14}
Reference PSD: {7} {15}
""".format(
                self.get_times(),
                self.get_traces(),
                self.get_frequencies(),
                self.get_spectrums(),
                self.get_avg_trace(),
                self.get_avg_psd(),
                self.get_reference(),
                self.get_reference_psd(),

                self.get_times().shape,
                self.get_traces().shape,
                self.get_avg_trace().shape,
                self.get_frequencies().shape,
                self.get_spectrums().shape,
                self.get_avg_psd().shape,
                self.get_reference().shape,
                self.get_reference_psd().shape
            )

            return s0 + s1 + s2 + s3 + s4
        except Exception as err:
            return str(err)

    def save_npz(self, folder):
        """
        Saves the traces to the NPZ file
        """
        filename = self.get_fullname() + "_new.npz"
        full_file = os.path.join(folder, filename)
        np.savez_compressed(full_file,
                            times=self.get_times(),
                            frequencies=self.get_frequencies(),
                            traces=self.get_traces(),
                            spectrums=self.get_spectrums(),
                            avg_trace=self.get_avg_trace(),
                            avg_psd=self.get_avg_psd(),
                            reference=self.get_reference(),
                            reference_psd=self.get_reference_psd()
        )
        return full_file

    def save_mat(self, folder):
        """
        Saves the traces to the MAT file
        """
        filename = self.get_fullname() + "_new.mat"
        full_file = os.path.join(folder, filename)
        mdict = {
            "animal_name": self.get_animal_name(),
            "prefix": self.get_prefix_name(),
            "postfix": self.get_postfix_name(),
            "case_name": self.get_case_name(),
            "roi_name": self.get_roi_name(),
            "fullname": self.get_fullname(),
            "times": self.get_times(),
            "frequencies": self.get_frequencies(),
            "traces": self.get_traces(),
            "spectrums": self.get_spectrums(),
            "avg_trace": self.get_avg_trace(),
            "avg_psd": self.get_avg_psd(),
            "reference": self.get_reference(),
            "reference_psd": self.get_reference_psd()
        }
        for key, value in self.__train_properties.items():
            mdict["TRA_" + key] = value
        for key, value in self.__synchronization_properties.items():
            mdict["SYN_" + key] = value
        for key, value in self.__isoline_properties.items():
            mdict["ISO_" + key] = value
        scipy.io.savemat(full_file, mdict)
        return full_file
