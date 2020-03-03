//
// Created by serik1987 on 18.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Acc_EXCEPTIONS_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Acc_EXCEPTIONS_H

extern "C" {

    static PyObject* PyImanA_AccumulatorError = NULL;
    static PyObject* PyImanA_NotAccumulatedError = NULL;
    static PyObject* PyImanA_BadPreprocessFilterRadiusError = NULL;
    static PyObject* PyImanA_InterruptedError = NULL;

    /**
     * Initializes exceptions of the accumulators package
     *
     * @param module pointer to the _imageanalysis module
     * @return 0 in success, -1 in failure
     */
    static int PyImanA_Exception_Init(PyObject* module){

        PyImanA_AccumulatorError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.accumulators.AccumulatorError",
                "This is the base class for all exceptions that will be originated in the accumulators package",
                PyIman_ImanError, NULL);
        if (PyModule_AddObject(module, "_accumulators_AccumulatorError", PyImanA_AccumulatorError) < 0){
            return -1;
        }

        PyImanA_NotAccumulatedError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.accumulators.NotAccumulatedError",
                "This error will be generated when you access to the property of the Accumulator object\n"
                "(or its derivative) that will be available after accumulation, not now",
                PyImanA_AccumulatorError, NULL);
        if (PyModule_AddObject(module, "_accumulators_NotAccumulatedError", PyImanA_NotAccumulatedError) < 0){
            return -1;
        }

        PyImanA_BadPreprocessFilterRadiusError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.accumulators.BadPreprocessFilterRadiusError",
                "When you adjust parameters of the preprocessor filter, their values shall be within the range from \n"
                "zero (not included) till 1/2 of the map size (included) and expressed in pixels\n"
                "If this is not truth, you will receive this error",
                PyImanA_BadPreprocessFilterRadiusError, NULL);
        if (PyModule_AddObject(module, "_accumulators_BadPreprocessFilterRadiusError",
                PyImanA_BadPreprocessFilterRadiusError) < 0){
            return -1;
        }

        PyImanA_InterruptedError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.accumulators.InterruptedError",
                "This error will be generated when the accumulation process is interrupted by the user",
                PyImanA_InterruptedError, NULL);
        if (PyModule_AddObject(module, "_accumulators_InterruptedError", PyImanA_InterruptedError) < 0){
            return -1;
        }

        return 0;
    }

    /**
     * Destroys all accumulator exceptions on fail
     */
    static void PyImanA_Exception_Destroy(){
        Py_XDECREF(PyImanA_AccumulatorError);
        Py_XDECREF(PyImanA_NotAccumulatedError);
        Py_XDECREF(PyImanA_BadPreprocessFilterRadiusError);
        Py_XDECREF(PyImanA_InterruptedError);
    }

    /**
     * Generates Python exception when C++ throws any exception belonging
     * to the accumulators class
     *
     * @param handle pointer to the C++ exception
     * @return 0 if the exception is not generated, -1 otherwise
     */
    static int PyImanA_Exception_process(const void* handle){
        using namespace GLOBAL_NAMESPACE;

        auto* exception_handle = (std::exception*)handle;
        auto* AccumulatorError_handle = dynamic_cast<Accumulator::AccumulatorException*>(exception_handle);

        if (AccumulatorError_handle != nullptr){
            auto* NotAccumulatedError_handle = dynamic_cast<Accumulator::NotAccumulatedException*>(exception_handle);
            if (NotAccumulatedError_handle != nullptr){
                PyErr_SetString(PyImanA_NotAccumulatedError, exception_handle->what());
                return -1;
            }

            auto* BadPreprocessFilterRadiusError_handle =
                    dynamic_cast<FrameAccumulator::BadPreprocessFilterRadiusException*>(exception_handle);
            if (BadPreprocessFilterRadiusError_handle != nullptr){
                PyErr_SetString(PyImanA_BadPreprocessFilterRadiusError, exception_handle->what());
                return -1;
            }

            auto* InterruptedError_handle =
                    dynamic_cast<Accumulator::InterruptedException*>(exception_handle);
            if (InterruptedError_handle != nullptr){
                PyErr_SetString(PyImanA_InterruptedError, exception_handle->what());
                return -1;
            }

            PyErr_SetString(PyImanA_AccumulatorError, exception_handle->what());
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Acc_EXCEPTIONS_H
