//
// Created by serik1987 on 08.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_COMPRESSOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_COMPRESSOR_H

extern "C" {

    static int PyImanC_Compressor_Create(PyObject *module) {
        printf("SO Compressor create\n");
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSOR_H
