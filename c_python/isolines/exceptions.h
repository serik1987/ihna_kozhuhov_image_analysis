//
// Created by serik1987 on 13.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_isolines_EXCEPTIONS_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_isolines_EXCEPTIONS_H

extern "C" {

    static int PyImanI_Exception_Create(PyObject* module){
        return 0;
    }

    static void PyImanI_Exception_Destroy(){

    }

    static int PyImanI_Exception_Process(const void* handle){
        auto* exception_handle = (std::exception*)handle;

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXCEPTIONS_H
