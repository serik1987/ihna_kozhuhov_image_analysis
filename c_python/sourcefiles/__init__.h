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
#include "../../cpp/source_files/StreamFileTrain.h"
#include "../../cpp/source_files/CompChunk.h"

extern "C" {
    static int PyImanS_TotalChunksAdded = 0;
    static const int PyImanS_TotalChunksExisted = 6;
    static PyObject* PyImanS_ChunkTypes[PyImanS_TotalChunksExisted];
};

#include "exceptions.h"
#include "FileTrain.h"
#include "StreamFileTrain.h"
#include "CompressedFileTrain.h"
#include "SourceFile.h"
#include "AnalysisSourceFile.h"
#include "GreenSourceFile.h"
#include "TrainSourceFile.h"
#include "StreamSourceFile.h"
#include "FileTrainIterator.h"
#include "StreamFileTrainIterator.h"
#include "CompressedSourceFile.h"
#include "CompressedFileTrainIterator.h"
#include "Chunk.h"
#include "SoftChunk.h"
#include "IsoiChunk.h"
#include "CompChunk.h"
#include "CostChunk.h"
#include "DataChunk.h"

#include "ChunkCreator.h"

extern "C" {

    static void PyImanS_DeleteSourceFileClasses(){
        Py_DECREF(&PyImanS_CompressedFileTrainIteratorType);
        Py_DECREF(&PyImanS_CompressedFileTrainType);
        Py_DECREF(&PyImanS_SourceFileTrainIteratorType);
        Py_DECREF(&PyImanS_FileTrainIteratorType);
        Py_DECREF(&PyImanS_StreamSourceFileType);
        Py_DECREF(&PyImanS_TrainSourceFileType);
        Py_DECREF(&PyImanS_GreenSourceFileType);
        Py_DECREF(&PyImanS_AnalysisSourceFileType);
        Py_DECREF(&PyImanS_SourceFileType);
        Py_DECREF(&PyImanS_CompressedFileTrainType);
        Py_DECREF(&PyImanS_StreamFileTrainType);
        Py_DECREF(&PyImanS_FileTrainType);
        PyImanS_Destroy_exceptions();
    }

    static void PyImanS_Destroy(){
        for (int i=0; i < PyImanS_TotalChunksAdded; ++i){
            Py_DECREF(PyImanS_ChunkTypes[i]);
        }
        PyImanS_DeleteSourceFileClasses();
    }


    static int PyImanS_Init(PyObject* imageanalysis){

        if (PyImanS_Create_exceptions(imageanalysis) < 0){
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_FileTrain_Create(imageanalysis) < 0){
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_StreamFileTrain_Create(imageanalysis) < 0){
            Py_DECREF(&PyImanS_FileTrainType);
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_CompressedFileTrain_Create(imageanalysis) < 0){
            Py_DECREF(&PyImanS_StreamFileTrainType);
            Py_DECREF(&PyImanS_FileTrainType);
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_SourceFile_Create(imageanalysis) < 0){
            Py_DECREF(&PyImanS_CompressedFileTrainType);
            Py_DECREF(&PyImanS_StreamFileTrainType);
            Py_DECREF(&PyImanS_FileTrainType);
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_AnalysisSourceFile_Create(imageanalysis) < 0){
            Py_DECREF(&PyImanS_SourceFileType);
            Py_DECREF(&PyImanS_CompressedFileTrainType);
            Py_DECREF(&PyImanS_StreamFileTrainType);
            Py_DECREF(&PyImanS_FileTrainType);
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_GreenSourceFile_Create(imageanalysis) < 0){
            Py_DECREF(&PyImanS_AnalysisSourceFileType);
            Py_DECREF(&PyImanS_SourceFileType);
            Py_DECREF(&PyImanS_CompressedFileTrainType);
            Py_DECREF(&PyImanS_StreamFileTrainType);
            Py_DECREF(&PyImanS_FileTrainType);
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_TrainSourceFile_Create(imageanalysis) < 0){
            Py_DECREF(&PyImanS_GreenSourceFileType);
            Py_DECREF(&PyImanS_AnalysisSourceFileType);
            Py_DECREF(&PyImanS_SourceFileType);
            Py_DECREF(&PyImanS_CompressedFileTrainType);
            Py_DECREF(&PyImanS_StreamFileTrainType);
            Py_DECREF(&PyImanS_FileTrainType);
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_StreamSourceFile_Create(imageanalysis) < 0){
            Py_DECREF(&PyImanS_TrainSourceFileType);
            Py_DECREF(&PyImanS_GreenSourceFileType);
            Py_DECREF(&PyImanS_AnalysisSourceFileType);
            Py_DECREF(&PyImanS_SourceFileType);
            Py_DECREF(&PyImanS_CompressedFileTrainType);
            Py_DECREF(&PyImanS_StreamFileTrainType);
            Py_DECREF(&PyImanS_FileTrainType);
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_FileTrainIterator_Create(imageanalysis) < 0){
            Py_DECREF(&PyImanS_StreamSourceFileType);
            Py_DECREF(&PyImanS_TrainSourceFileType);
            Py_DECREF(&PyImanS_GreenSourceFileType);
            Py_DECREF(&PyImanS_AnalysisSourceFileType);
            Py_DECREF(&PyImanS_SourceFileType);
            Py_DECREF(&PyImanS_CompressedFileTrainType);
            Py_DECREF(&PyImanS_StreamFileTrainType);
            Py_DECREF(&PyImanS_FileTrainType);
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_StreamFileTrainIterator_Create(imageanalysis) < 0){
            Py_DECREF(&PyImanS_FileTrainIteratorType);
            Py_DECREF(&PyImanS_StreamSourceFileType);
            Py_DECREF(&PyImanS_TrainSourceFileType);
            Py_DECREF(&PyImanS_GreenSourceFileType);
            Py_DECREF(&PyImanS_AnalysisSourceFileType);
            Py_DECREF(&PyImanS_SourceFileType);
            Py_DECREF(&PyImanS_CompressedFileTrainType);
            Py_DECREF(&PyImanS_StreamFileTrainType);
            Py_DECREF(&PyImanS_FileTrainType);
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_CompressedSourceFile_Create(imageanalysis) < 0){
            Py_DECREF(&PyImanS_SourceFileTrainIteratorType);
            Py_DECREF(&PyImanS_FileTrainIteratorType);
            Py_DECREF(&PyImanS_StreamSourceFileType);
            Py_DECREF(&PyImanS_TrainSourceFileType);
            Py_DECREF(&PyImanS_GreenSourceFileType);
            Py_DECREF(&PyImanS_AnalysisSourceFileType);
            Py_DECREF(&PyImanS_SourceFileType);
            Py_DECREF(&PyImanS_CompressedFileTrainType);
            Py_DECREF(&PyImanS_StreamFileTrainType);
            Py_DECREF(&PyImanS_FileTrainType);
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_CompressedFileTrainIterator_Create(imageanalysis) < 0){
            Py_DECREF(&PyImanS_CompressedFileTrainType);
            Py_DECREF(&PyImanS_SourceFileTrainIteratorType);
            Py_DECREF(&PyImanS_FileTrainIteratorType);
            Py_DECREF(&PyImanS_StreamSourceFileType);
            Py_DECREF(&PyImanS_TrainSourceFileType);
            Py_DECREF(&PyImanS_GreenSourceFileType);
            Py_DECREF(&PyImanS_AnalysisSourceFileType);
            Py_DECREF(&PyImanS_SourceFileType);
            Py_DECREF(&PyImanS_CompressedFileTrainType);
            Py_DECREF(&PyImanS_StreamFileTrainType);
            Py_DECREF(&PyImanS_FileTrainType);
            PyImanS_Destroy_exceptions();
            return -1;
        }

        if (PyImanS_Chunk_Create(imageanalysis) < 0){
            PyImanS_DeleteSourceFileClasses();
            return -1;
        }

        if (PyImanS_SoftChunk_Create(imageanalysis) < 0){
            PyImanS_Destroy();
            return -1;
        }

        if (PyImanS_IsoiChunk_Create(imageanalysis) < 0){
            PyImanS_Destroy();
            return -1;
        }

        if (PyImanS_CompChunk_Create(imageanalysis) < 0){
            PyImanS_Destroy();
            return -1;
        }

        if (PyImanS_CostChunk_Create(imageanalysis) < 0){
            PyImanS_Destroy();
            return -1;
        }

        if (PyImanS_DataChunk_Create(imageanalysis) < 0){
            PyImanS_Destroy();
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS___INIT___H
