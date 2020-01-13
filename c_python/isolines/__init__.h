//
// Created by serik1987 on 13.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS___isolines_INIT___H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS___isolines_INIT___H

#include "../../cpp/isolines/Isoline.h"
#include "../../cpp/isolines/NoIsoline.h"
#include "../../cpp/isolines/LinearFitIsoline.h"
#include "../../cpp/isolines/TimeAverageIsoline.h"

extern "C" {
    PyTypeObject* PyImanI_Isoline_Handle = NULL;
    PyTypeObject* PyImanI_NoIsoline_Handle = NULL;
    PyTypeObject* PyImanI_LinearFitIsoline_Handle = NULL;
    PyTypeObject* PyImanI_TimeAverageIsoline_Handle = NULL;
};

#include "exceptions.h"
#include "isoline.h"
#include "NoIsoline.h"
#include "LinearFitIsoline.h"
#include "TimeAverageIsoline.h"

extern "C"{

    static void PyImanI_Destroy(){
        Py_XDECREF(PyImanI_TimeAverageIsoline_Handle);
        Py_XDECREF(PyImanI_LinearFitIsoline_Handle);
        Py_XDECREF(PyImanI_NoIsoline_Handle);
        Py_XDECREF(PyImanI_Isoline_Handle);
        PyImanI_Exception_Destroy();
    }

    static int PyImanI_Init(PyObject* module){

        if (PyImanI_Exception_Create(module) < 0){
            PyImanI_Destroy();
            return -1;
        }

        if (PyImanI_Isoline_Create(module) < 0){
            PyImanI_Destroy();
            return -1;
        }

        if (PyImanI_NoIsoline_Create(module) < 0){
            PyImanI_Destroy();
            return -1;
        }

        if (PyImanI_LinearFitIsoline_Create(module) < 0){
            PyImanI_Destroy();
            return -1;
        }

        if (PyImanI_TimeAverageIsoline_Create(module) < 0){
            PyImanI_Destroy();
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS___INIT___H
