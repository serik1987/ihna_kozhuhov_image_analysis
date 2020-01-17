//
// Created by serik1987 on 13.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_isolines_EXCEPTIONS_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_isolines_EXCEPTIONS_H

extern "C" {

    static PyObject* PyImanI_IsolineError = NULL;
    static PyObject* PyImanI_AverageCyclesError = NULL;

    static int PyImanI_Exception_Create(PyObject* module){
        PyImanI_IsolineError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov,imageanalysis.isolines.IsolineError",
                "Base error for all isolines",
                PyIman_ImanError, NULL);
        if (PyModule_AddObject(module, "_isolines_IsolineError", PyImanI_IsolineError) < 0){
            return -1;
        }

        PyImanI_AverageCyclesError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.isolines.AverageCyclesError",
                "This error will be generated when number of average cycles is zero or negative",
                PyImanI_AverageCyclesError, NULL);
        if (PyModule_AddObject(module, "_isolines_AverageCyclesError", PyImanI_AverageCyclesError) < 0){
            return -1;
        }

        return 0;
    }

    static void PyImanI_Exception_Destroy(){
        Py_XDECREF(PyImanI_IsolineError);
        Py_XDECREF(PyImanI_AverageCyclesError);
    }

    static int PyImanI_Exception_Process(const void* handle){
        using namespace GLOBAL_NAMESPACE;
        auto* exception_handle = (std::exception*)handle;
        auto* isoline_handle = dynamic_cast<Isoline::IsolineException*>(exception_handle);

        if (isoline_handle != nullptr){
            auto* avg_cycles_error = dynamic_cast<TimeAverageIsoline::AverageCyclesException*>(isoline_handle);

            if (avg_cycles_error != nullptr){
                PyErr_SetString(PyImanI_AverageCyclesError, avg_cycles_error->what());
            } else {
                PyErr_SetString(PyImanI_IsolineError, isoline_handle->what());
            }
            return -1;
        } else {
            return 0;
        }
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXCEPTIONS_H
