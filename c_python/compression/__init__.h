//
// Created by serik1987 on 08.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS___compression__INIT___H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS___compression__INIT___H

extern "C" {

    #include "../../cpp/compression/Compressor.h"
    #include "../../cpp/compression/Decompressor.h"

#define PyImanC_CLASS_NUMBER 2
    static int PyImanC_Current_class = 0;
    static PyObject* PyImanC_Class_list[PyImanC_CLASS_NUMBER];

    #include "exceptions.h"
    #include "compressor.h"
    #include "decompressor.h"

    static void PyImanC_Destroy() {
        PyImanC_Destroy_exceptions();
        for (int i=0; i < PyImanC_CLASS_NUMBER; ++i){
            Py_XDECREF(PyImanC_Class_list[i]);
        }
    }

    static int PyImanC_Init(PyObject *module) {

        for (int i=0; i < PyImanC_CLASS_NUMBER; ++i){
            PyImanC_Class_list[i] = NULL;
        }

        if (PyImanC_Init_exceptions(module) < 0) {
            PyImanC_Destroy();
            return -1;
        }

        if (PyImanC_Compressor_Create(module) < 0) {
            PyImanC_Destroy();
            return -1;
        }

        if (PyImanC_Decompressor_Create(module) < 0) {
            PyImanC_Destroy();
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS___compression__INIT___H
