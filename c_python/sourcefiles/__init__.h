//
// Created by serik1987 on 21.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS___INIT_SOURCE_FILES___H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS___INIT_SOURCE_FILES___H

#include "../../cpp/source_files/FileTrain.h"
#include "../../cpp/source_files/CompressedFileTrain.h"
#include "../../cpp/source_files/AnalysisSourceFile.h"
#include "../../cpp/source_files/GreenSourceFile.h"
#include "../../cpp/source_files/StreamSourceFile.h"
#include "../../cpp/source_files/DataChunk.h"

#include "exceptions.h"

extern "C" {


    static int PyImanS_Init(PyObject* imageanalysis){

        if (PyImanS_Create_exceptions(imageanalysis) < 0){
            PyImanS_Destroy_exceptions();
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS___INIT___H
