//
// Created by serik1987 on 19.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SOURCE_FILES_EXCEPTIONS_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SOURCE_FILES_EXCEPTIONS_H

#include "core.h"
#include "../../cpp/exceptions.h"
#include "../../cpp/source_files/FileTrain.h"
#include "../../cpp/source_files/AnalysisSourceFile.h"
#include "../../cpp/source_files/GreenSourceFile.h"
#include "../../cpp/source_files/StreamSourceFile.h"
#include "../../cpp/source_files/CompressedSourceFile.h"
#include "../../cpp/source_files/DataChunk.h"

#define FULL_ERROR_NAME_PREFIX FULL_NAME_PREFIX



#ifdef __cplusplus
extern "C" {
#endif


#define C_API_Exception_process 0
#define C_API_ImanError 1
#define C_API_IoError 2
#define C_API_TrainError 3
#define C_API_ExperimentModeError 4
#define C_API_SynchronizationChannelNumberError 5
#define C_API_UnsupportedExperimentModeError 6
#define C_API_SourceFileError 7
#define C_API_FrameNumberError 8
#define C_API_DataChunkNotFoundError 9
#define C_API_IsoiChunkSizeMismatchError 10
#define C_API_FileSizeMismatchError 11
#define C_API_ExperimentChunkMismatchError 12
#define C_API_FileHeaderMismatchError 13
#define C_API_FrameHeaderMismatchError 14
#define C_API_FrameDimensionsMismatchError 15
#define C_API_DataTypeMismatchError 16
#define C_API_CompChunkNotExistError 17
#define C_API_FileNotOpenedError 18
#define C_API_FileOpenError 19
#define C_API_FileReadError 20
#define C_API_ChunkError 21
#define C_API_UnsupportedChunkError 22
#define C_API_ChunkSizeMismatchError 23
#define C_API_ChunkNotFoundError 24
#define C_API_IsoiChunkNotFoundError 25
#define C_API_FileNotLoadedError 26
#define C_API_DataChunkSizeMismatchError 27
#define C_API_NotAnalysisFileError 28
#define C_API_NotGreenFileError 29
#define C_API_NotCompressedFileError 30
#define C_API_NotStreamFileError 31
#define C_API_NotTrainHeadError 32
#define C_API_INSTANCE_NUMBER 33

#ifdef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SOURCE_FILES_EXCEPTIONS_MODULE

    /* Function prototype */
    static int Exception_process(void* exception);

#else

    static void** PyImanS(_exceptions_API);
#define API_TABLE PyImanS(_exceptions_API)

#define PyImanS_Exception_process (*(int (*)(void*))PyImanS(_exceptions_API)[C_API_Exception_process])
#define Ihna__Kozhukhov__Image_analysis__Source_files_Exception_process PyImanS_Exception_process

#define PyImanS_ImanError (PyObject*)API_TABLE[C_API_ImanError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_ImanError PyImanS_ImanError


#define PyImanS_IoError (PyObject*)API_TABLE[C_API_IoError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_IoError PyImanS_IoError


#define PyImanS_TrainError (PyObject*)API_TABLE[C_API_TrainError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_TrainError PyImanS_TrainError


#define PyImanS_ExperimentModeError (PyObject*)API_TABLE[C_API_ExperimentModeError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_ExperimentModeError PyImanS_ExperimentModeError


#define PyImanS_SynchronizationChannelNumberError (PyObject*)API_TABLE[C_API_SynchronizationChannelNumberError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_SynchronizationChannelNumberError PyImanS_SynchronizationChannelNumberError


#define PyImanS_UnsupportedExperimentModeError (PyObject*)API_TABLE[C_API_UnsupportedExperimentModeError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_UnsupportedExperimentModeError\
PyImanS_UnsupportedExperimentModeError


#define PyImanS_SourceFileError (PyObject*)API_TABLE[C_API_SourceFileError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_SourceFileError PyImanS_SourceFileError


#define PyImanS_FrameNumberError (PyObject*)API_TABLE[C_API_FrameNumberError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_FrameNumberError PyImanS_FrameNumberError


#define PyImanS_DataChunkNotFoundError (PyObject*)API_TABLE[C_API_DataChunkNotFoundError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_DataChunkNotFoundError PyImanS_DataChunkNotFoundError


#define PyImanS_IsoiChunkSizeMismatchError (PyObject*)API_TABLE[C_API_IsoiChunkSizeMismatchError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_IsoiChunkSizeMismatchError PyImanS_IsoiChunkSizeMismatchError


#define PyImanS_FileSizeMismatchError (PyObject*)API_TABLE[C_API_FileSizeMismatchError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_FileSizeMismatchError PyImanS_FileSizeMismatchError


#define PyImanS_ExperimentChunkMismatchError (PyObject*)API_TABLE[C_API_ExperimentChunkMismatchError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_ExperimentChunkMismatchError PyImanS_ExperimentChunkMismatchError


#define PyImanS_FileHeaderMismatchError (PyObject*)API_TABLE[C_API_FileHeaderMismatchError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_FileHeaderMismatchError PyImanS_FileHeaderMismatchError


#define PyImanS_FrameHeaderMismatchError (PyObject*)API_TABLE[C_API_FrameHeaderMismatchError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_FrameHeaderMismatchError PyImanS_FrameHeaderMismatchError


#define PyImanS_FrameDimensionsMismatchError (PyObject*)API_TABLE[C_API_FrameDimensionsMismatchError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_FrameDimensionsMismatchError PyImanS_FrameDimensionsMismatchError


#define PyImanS_DataTypeMismatchError (PyObject*)API_TABLE[C_API_DataTypeMismatchError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_DataTypeMismatchError PyImanS_DataTypeMismatchError


#define PyImanS_CompChunkNotExistError (PyObject*)API_TABLE[C_API_CompChunkNotExistError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_CompChunkNotExistError PyImanS_CompChunkNotExistError


#define PyImanS_FileNotOpenedError (PyObject*)API_TABLE[C_API_FileNotOpenedError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_FileNotOpenedError PyImanS_FileNotOpenedError


#define PyImanS_FileOpenError (PyObject*)API_TABLE[C_API_FileOpenError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_FileOpenError PyImanS_FileOpenError


#define PyImanS_FileReadError (PyObject*)API_TABLE[C_API_FileReadError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_FileReadError PyImanS_FileReadError


#define PyImanS_ChunkError (PyObject*)API_TABLE[C_API_ChunkError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_ChunkError PyImanS_ChunkError


#define PyImanS_UnsupportedChunkError (PyObject*)API_TABLE[C_API_UnsupportedChunkError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_UnsupportedChunkError PyImanS_UnsupportedChunkError


#define PyImanS_ChunkSizeMismatchError (PyObject*)API_TABLE[C_API_ChunkSizeMismatchError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_ChunkSizeMismatchError PyImanS_ChunkSizeMismatchError


#define PyImanS_ChunkNotFoundError (PyObject*)API_TABLE[C_API_ChunkNotFoundError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_ChunkNotFoundError PyImanS_ChunkNotFoundError


#define PyImanS_IsoiChunkNotFoundError (PyObject*)API_TABLE[C_API_IsoiChunkNotFoundError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_IsoiChunkNotFoundError PyImanS_IsoiChunkNotFoundError


#define PyImanS_FileNotLoadedError (PyObject*)API_TABLE[C_API_FileNotLoadedError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_FileNotLoadedError PyImanS_FileNotLoadedError


#define PyImanS_DataChunkSizeMismatchError (PyObject*)API_TABLE[C_API_DataChunkSizeMismatchError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_DataChunkSizeMismatchError PyImanS_DataChunkSizeMismatchError


#define PyImanS_NotAnalysisFileError (PyObject*)API_TABLE[C_API_NotAnalysisFileError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_NotAnalysisFileError PyImanS_NotAnalysisFileError


#define PyImanS_NotGreenFileError (PyObject*)API_TABLE[C_API_NotGreenFileError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_NotGreenFileError PyImanS_NotGreenFileError


#define PyImanS_NotCompressedFileError (PyObject*)API_TABLE[C_API_NotCompressedFileError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_NotCompressedFileError PyImanS_NotCompressedFileError


#define PyImanS_NotStreamFileError (PyObject*)API_TABLE[C_API_NotStreamFileError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_NotStreamFileError PyImanS_NotStreamFileError


#define PyImanS_NotTrainHeadError (PyObject*)API_TABLE[C_API_NotTrainHeadError]
#define Ihna__Kozhukhov__Image_analysis__Source_files_NotTrainHeadError PyImanS_NotTrainHeadError


    static int PyImanS(Import__exceptions)(void){
        PyImanS(_exceptions_API) = (void**)PyCapsule_Import(FULL_ERROR_NAME_PREFIX"_exceptions._c_API", 0);
        if (PyImanS(_exceptions_API) == NULL){
            return -1;
        }

        return 0;
    }
#define PyImanS_Import__exceptions PyImanS(Import__exceptions)

#endif

#ifdef __cplusplus
};
#endif

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXCEPTIONS_H
