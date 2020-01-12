//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS___Synchronization_INIT___H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS___Synchronization_INIT___H

#include "../../cpp/synchronization/Synchronization.h"
#include "../../cpp/synchronization/ExternalSynchronization.h"
#include "../../cpp/synchronization/NoSynchronization.h"

extern "C" {
    static PyObject *PyImanY_Synchronization_Handle = NULL;
    static PyObject* PyImanY_ExternalSynchronization_Handle = NULL;
    static PyObject* PyImanY_NoSynchronization_Handle = NULL;
}

#include "exceptions.h"
#include "Synchronization.h"
#include "ExternalSynchronization.h"
#include "NoSynchronization.h"

extern "C" {

    static void PyImanY_Destroy(){
        PyImanY_Exception_Destroy();
        Py_XDECREF(PyImanY_Synchronization_Handle);
        Py_XDECREF(PyImanY_ExternalSynchronization_Handle);
        Py_XDECREF(PyImanY_NoSynchronization_Handle);
    }

    static int PyImanY_Init(PyObject* module){

        if (PyImanY_Exception_Init(module) < 0){
            PyImanY_Destroy();
            return -1;
        }

        if (PyImanY_Synchronization_Create(module) < 0){
            PyImanY_Destroy();
            return -1;
        }

        if (PyImanY_ExternalSynchronization_Create(module) < 0){
            PyImanY_Destroy();
            return -1;
        }

        if (PyImanY_NoSynchronization_Create(module) < 0){
            PyImanY_Destroy();
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS___INIT___H
