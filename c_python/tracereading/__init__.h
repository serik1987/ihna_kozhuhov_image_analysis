//
// Created by serik1987 on 11.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS___tracereading_INIT___H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS___tracereading_INIT___H


extern "C" {

    static PyObject* PyImanT_TraceReaderTypeHandle = NULL;

#include "TraceReader.h"

    static void PyImanT_Destroy(){
        Py_XDECREF(PyImanT_TraceReaderTypeHandle);
    }

    static int PyImanT_Init(PyObject* module){

        if (PyImanT_TraceReader_Create(module) < 0){
            PyImanT_Destroy();
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS___INIT___H
