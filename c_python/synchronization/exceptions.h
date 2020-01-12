//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Y_EXCEPTIONS_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Y_EXCEPTIONS_H

extern "C" {

    static int PyImanY_Exception_Init(PyObject* module){
        return 0;
    }

    static void PyImanY_Exception_Destroy(){

    }

    static int PyImanY_Exception_process(const void* handle){
        auto* exception_handle = (std::exception*)handle;

        printf("SO synchronization package: exception process\n");

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXCEPTIONS_H
