//
// Created by serik1987 on 18.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS___Acc_INIT___H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS___Acc_INIT___H

#include "../../cpp/accumulators/Accumulator.h"
#include "../../cpp/accumulators/FrameAccumulator.h"
#include "../../cpp/accumulators/MapPlotter.h"
#include "../../cpp/accumulators/MapFilter.h"
#include "../../cpp/accumulators/TraceAutoReader.h"

extern "C" {

    static PyObject *PyImanA_Accumulator_Handle = NULL;
    static PyObject* PyImanA_FrameAccumulator_Handle = NULL;
    static PyObject* PyImanA_MapPlotter_Handle = NULL;
    static PyObject* PyImanA_MapFilter_Handle = NULL;
    static PyObject* PyImanA_TraceAutoReader_Handle = NULL;

}

#include "exceptions.h"
#include "Accumulator.h"
#include "FrameAccumulator.h"
#include "MapPlotter.h"
#include "MapFilter.h"
#include "TraceAutoReader.h"

extern "C" {

    /**
     * Destroys all _imageanalysis objects to be imported into accumulators package
     */
    static void PyImanA_Destroy(){
        PyImanA_Exception_Destroy();
        Py_XDECREF(PyImanA_Accumulator_Handle);
        Py_XDECREF(PyImanA_FrameAccumulator_Handle);
        Py_XDECREF(PyImanA_MapPlotter_Handle);
        Py_XDECREF(PyImanA_MapFilter_Handle);
    }

    /**
     * Initializes all module objects that will be imported to the accumulators package
     *
     * @param module reference to the imageanalysis module
     * @return 0 on suceess, -1 on failure
     */
    static int PyImanA_Init(PyObject* module){

        if (PyImanA_Exception_Init(module) < 0) {
            PyImanA_Destroy();
            return -1;
        }

        if (PyImanA_Accumulator_Create(module) < 0){
            PyImanA_Destroy();
            return -1;
        }

        if (PyImanA_FrameAccumulator_Create(module) < 0){
            PyImanA_Destroy();
            return -1;
        }

        if (PyImanA_MapPlotter_Create(module) < 0){
            PyImanA_Destroy();
            return -1;
        }

        if (PyImanA_MapFilter_Create(module) < 0){
            PyImanA_Destroy();
            return -1;
        }

        if (PyImanA_TraceAutoReader_Create(module) < 0){
            PyImanA_Destroy();
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS___Acc_INIT___H
