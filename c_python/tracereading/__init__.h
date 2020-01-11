//
// Created by serik1987 on 11.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS___tracereading_INIT___H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS___tracereading_INIT___H

#include "../../cpp/tracereading/TraceReader.h"


extern "C" {

    static PyObject *PyImanT_TraceReaderTypeHandle = NULL;
}

#include "exceptions.h"
#include "TraceReader.h"

extern "C" {

    static void PyImanT_Destroy(){
        PyImanT_Exceptions_Destroy();
        Py_XDECREF(PyImanT_TraceReaderTypeHandle);
    }

    static int PyImanT_Init(PyObject* module){

        if (PyImanT_Exceptions_Create(module) < 0){
            PyImanT_Destroy();
            return -1;
        }

        if (PyImanT_TraceReader_Create(module) < 0){
            PyImanT_Destroy();
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS___INIT___H
