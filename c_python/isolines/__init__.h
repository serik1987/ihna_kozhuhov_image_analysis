//
// Created by serik1987 on 13.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS___isolines_INIT___H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS___isolines_INIT___H

#include "../../cpp/isolines/Isoline.h"

extern "C" {
    PyTypeObject* PyImanI_Isoline_Handle = NULL;
};

#include "exceptions.h"
#include "isoline.h"

extern "C"{

    static void PyImanI_Destroy(){
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

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS___INIT___H
