//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Y_EXCEPTIONS_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Y_EXCEPTIONS_H

extern "C" {

    static PyObject* PyImanY_SynchronizationError = NULL;
    static PyObject* PyImanY_FileNotOpenedError = NULL;
    static PyObject* PyImanY_NotSynchronizedError = NULL;
    static PyObject* PyImanY_FrameRangeError = NULL;
    static PyObject* PyImanY_StimulusPeriodError = NULL;
    static PyObject* PyImanY_InitialCycleError = NULL;
    static PyObject* PyImanY_FinalCycleError = NULL;
    static PyObject* PyImanY_SynchronizationChannelError = NULL;
    static PyObject* PyImanY_NoSignalDetectedError = NULL;
    static PyObject* PyImanY_TooFewFramesError = NULL;
    static PyObject* PyImanY_BadHarmonicError = NULL;
    static PyObject* PyImanY_NoisySignalError = NULL;
    static PyObject* PyImanY_BadThresholdError = NULL;


    static int PyImanY_Exception_Init(PyObject* module){
        PyImanY_SynchronizationError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.synchronization.SynchronizationError",
                "Base class for all errors that will occur during the synchronization",
                PyIman_ImanError, NULL);
        if (PyModule_AddObject(module, "_synchronization_SynchronizationError", PyImanY_SynchronizationError) < 0){
            return -1;
        }

        PyImanY_FileNotOpenedError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.synchronization.FileNotOpenedError",
                "This error will be generated when you create a synchronization object and will base the \n"
                "StreamFileTrain as an input argument but the train has not been open()'ed.",
                PyImanY_SynchronizationError, NULL);
        if (PyModule_AddObject(module, "_synchronization_FileNotOpenedError", PyImanY_FileNotOpenedError) < 0){
            return -1;
        }

        PyImanY_NotSynchronizedError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.synchronization.NotSynchronizedError",
                "This error will be generated when you try to read some property of the Synchronization object \n"
                "but the value of this property is not available because you did not synchronize the object\n",
                PyImanY_SynchronizationError, NULL);
        if (PyModule_AddObject(module, "_synchronization_NotSynchronizedError", PyImanY_NotSynchronizedError) < 0){
            return -1;
        }

        PyImanY_FrameRangeError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.synchronization.FrameRangeError",
                "This error will be generated when you set the frame range for NoSynchronization that doesn't lie \n"
                "within the range of frames available within the record",
                PyImanY_SynchronizationError, NULL);
        if (PyModule_AddObject(module, "_synchronization_FrameRangeError", PyImanY_FrameRangeError) < 0){
            return -1;
        }

        PyImanY_StimulusPeriodError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.synchronization.StimulusPeriodError",
                "This error will be generated when you set the stimulus period value that is either non-positive or \n"
                "exceed the total number of frames",
                PyImanY_SynchronizationError, NULL);
        if (PyModule_AddObject(module, "_synchronization_StimulusPeriodError", PyImanY_StimulusPeriodError) < 0){
            return -1;
        }

        PyImanY_InitialCycleError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.synchronization.InitialCycleError",
                "This error will be generated when you set the impossible value of the analysis initial cycle",
                PyImanY_SynchronizationError, NULL);
        if (PyModule_AddObject(module, "_synchronization_InitialCycleError", PyImanY_InitialCycleError) < 0){
            return -1;
        }

        PyImanY_FinalCycleError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.synchronization.FinalCycleError",
                "This error will be generated when you set the impossible value of the final cycle",
                PyImanY_SynchronizationError, NULL);
        if (PyModule_AddObject(module, "_synchronization_FinalCycleError", PyImanY_FinalCycleError) < 0){
            return -1;
        }

        PyImanY_SynchronizationChannelError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.synchronization.SynchronizationChannelError",
                "This error will be generated when the channel_number property of the ExternalSynchronization\n"
                "is set to non-existent channel",
                PyImanY_SynchronizationError, NULL);
        if (PyModule_AddObject(module, "_synchronization_SynchronizationChannelError",
                PyImanY_SynchronizationChannelError) < 0){
            return -1;
        }

        PyImanY_NoSignalDetectedError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.synchronization.NoSignalDetectedError",
                "This error will be generated when you selected external synchronization and used a synchronization \n"
                "channel that doesn't contain any signal",
                PyImanY_SynchronizationError, NULL);
        if (PyModule_AddObject(module, "_synchronization_NoSignalDetectedError", PyImanY_NoSignalDetectedError) < 0){
            return -1;
        }

        PyImanY_TooFewFramesError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.synchronization.TooFewFramesError",
                "If this error is generated this means that the record is no longer suitable for the external \n"
                "synchronization",
                PyImanY_SynchronizationError, NULL);
        if (PyModule_AddObject(module, "_synchronization_TooFewFramesError", PyImanY_TooFewFramesError) < 0){
            return -1;
        }

        PyImanY_BadHarmonicError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sychronization.BadHarmonicErrpr",
                "This error is generated when you selected a harmonic value that is not suitable for the precise \n"
                "analysis",
                PyImanY_SynchronizationError, NULL);
        if (PyModule_AddObject(module, "_synchronization_BadHarmonicError", PyImanY_BadHarmonicError) < 0){
            return -1;
        }

        PyImanY_NoisySignalError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.synchronization.NoisySignalError",
                "This error is generated when the synchronization signal is bad or corrupted",
                PyImanY_SynchronizationError, NULL
                );
        if (PyModule_AddObject(module, "_synchronization_NoisySignalError", PyImanY_NoisySignalError) < 0){
            return -1;
        }

        PyImanY_BadThresholdError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.synchronization.BadThresholdError",
                "This error is generated when you set a bad value of the external synchronization threshold",
                PyImanY_SynchronizationError, NULL);
        if (PyModule_AddObject(module, "_synchronization_BadThresholdError", PyImanY_BadThresholdError) < 0){
            return -1;
        }

        return 0;
    }

    static void PyImanY_Exception_Destroy(){
        Py_XDECREF(PyImanY_SynchronizationError);
        Py_XDECREF(PyImanY_FileNotOpenedError);
        Py_XDECREF(PyImanY_NotSynchronizedError);
        Py_XDECREF(PyImanY_FrameRangeError);
        Py_XDECREF(PyImanY_StimulusPeriodError);
        Py_XDECREF(PyImanY_InitialCycleError);
        Py_XDECREF(PyImanY_FinalCycleError);
        Py_XDECREF(PyImanY_NoSignalDetectedError);
        Py_XDECREF(PyImanY_TooFewFramesError);
        Py_XDECREF(PyImanY_BadHarmonicError);
        Py_XDECREF(PyImanY_NoisySignalError);
        Py_XDECREF(PyImanY_BadThresholdError);
    }

    static int PyImanY_Exception_process(const void* handle){
        using namespace GLOBAL_NAMESPACE;
        auto* exception_handle = (std::exception*)handle;
        auto* synchronization_handle = dynamic_cast<Synchronization::SynchronizationException*>(exception_handle);

        if (synchronization_handle != nullptr){
            auto* file_not_opened_handle =
                    dynamic_cast<Synchronization::FileNotOpenedException*>(synchronization_handle);
            auto* not_synchronized_handle =
                    dynamic_cast<Synchronization::NotSynchronizedException*>(synchronization_handle);
            auto* frame_range_handle =
                    dynamic_cast<NoSynchronization::FrameRangeException*>(synchronization_handle);
            auto* stimulus_period_handle =
                    dynamic_cast<QuasiStimulusSynchronization::StimulusPeriodException*>(synchronization_handle);
            auto* initial_cycle_handle =
                    dynamic_cast<QuasiStimulusSynchronization::InitialCycleException*>(synchronization_handle);
            auto* final_cycle_handle =
                    dynamic_cast<QuasiStimulusSynchronization::FinalCycleException*>(synchronization_handle);
            auto* synchronization_channel_handle =
                    dynamic_cast<ExternalSynchronization::SynchronizationChannelException*>(synchronization_handle);
            auto* no_signal_detected_handle =
                    dynamic_cast<ExternalSynchronization::NoSignalException*>(synchronization_handle);
            auto* too_few_frames_handle =
                    dynamic_cast<ExternalSynchronization::TooFewFramesException*>(synchronization_handle);
            auto* bad_harmonic_handle =
                    dynamic_cast<Synchronization::BadHarmonicException*>(synchronization_handle);
            auto* noisy_signal_handle =
                    dynamic_cast<ExternalSynchronization::NoisySignalException*>(synchronization_handle);
            auto* bad_threshold_handle =
                    dynamic_cast<ExternalSynchronization::BadThresholdException*>(synchronization_handle);

            if (file_not_opened_handle != nullptr) {
                PyErr_SetString(PyImanY_FileNotOpenedError, file_not_opened_handle->what());
                return -1;
            } else if (not_synchronized_handle != nullptr) {
                PyErr_SetString(PyImanY_NotSynchronizedError, not_synchronized_handle->what());
                return -1;
            } else if (frame_range_handle != nullptr) {
                PyErr_SetString(PyImanY_FrameRangeError, frame_range_handle->what());
                return -1;
            } else if (stimulus_period_handle != nullptr) {
                PyErr_SetString(PyImanY_StimulusPeriodError, stimulus_period_handle->what());
                return -1;
            } else if (initial_cycle_handle != nullptr) {
                PyErr_SetString(PyImanY_InitialCycleError, initial_cycle_handle->what());
                return -1;
            } else if (final_cycle_handle != nullptr) {
                PyErr_SetString(PyImanY_FinalCycleError, final_cycle_handle->what());
                return -1;
            } else if (synchronization_channel_handle != nullptr) {
                PyErr_SetString(PyImanY_SynchronizationChannelError, synchronization_channel_handle->what());
                return -1;
            } else if (no_signal_detected_handle != nullptr) {
                PyErr_SetString(PyImanY_NoSignalDetectedError, no_signal_detected_handle->what());
                return -1;
            } else if (too_few_frames_handle != nullptr) {
                PyErr_SetString(PyImanY_TooFewFramesError, too_few_frames_handle->what());
                return -1;
            } else if (bad_harmonic_handle != nullptr) {
                PyErr_SetString(PyImanY_BadHarmonicError, bad_harmonic_handle->what());
                return -1;
            } else if (noisy_signal_handle != nullptr) {
                PyErr_SetString(PyImanY_NoisySignalError, noisy_signal_handle->what());
                return -1;
            } else if (bad_threshold_handle != nullptr) {
                PyErr_SetString(PyImanY_BadThresholdError, bad_threshold_handle->what());
                return -1;
            } else {
                PyErr_SetString(PyImanY_SynchronizationError, synchronization_handle->what());
                return -1;
            }
        } else {
            return 0;
        }
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXCEPTIONS_H
