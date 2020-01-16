//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Y_EXCEPTIONS_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Y_EXCEPTIONS_H

extern "C" {

    static PyObject* PyImanY_SynchronizationError = NULL;
    static PyObject* PyImanY_FileNotOpenedError = NULL;
    static PyObject* PyImanY_NotSynchronizedError = NULL;

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

        return 0;
    }

    static void PyImanY_Exception_Destroy(){
        Py_XDECREF(PyImanY_SynchronizationError);
        Py_XDECREF(PyImanY_FileNotOpenedError);
        Py_XDECREF(PyImanY_NotSynchronizedError);
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

            if (file_not_opened_handle != nullptr) {
                PyErr_SetString(PyImanY_FileNotOpenedError, file_not_opened_handle->what());
                return -1;
            } else if (not_synchronized_handle != nullptr){
                PyErr_SetString(PyImanY_NotSynchronizedError, not_synchronized_handle->what());
                return -1;
            } else {
                PyErr_SetString(PyImanY_SynchronizationError, synchronization_handle->what());
                return -1;
            }
        } else {
            return 0;
        }
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXCEPTIONS_H
